import { create } from 'zustand';
import { forensicApi } from '../services/api';
import useToastStore from './useToastStore';

const createAnalysisSlice = () => ({
  isAnalyzing: false,
  results: null,
  error: null,
});

/**
 * Normalize an API envelope response into a flat results object.
 *
 * The API returns `{ success, data: { risk_score, ... }, error }`.
 * Previously, stores saved the raw envelope and components had to guess
 * between `results.risk_percentage`, `results.data?.risk_percent`, etc.
 *
 * Now the store always unwraps `data` so components get a flat shape:
 *   results.risk_percent, results.verdict, results.model_scores, ...
 */
function normalizeResults(apiResponse) {
  if (!apiResponse) return null;

  // If the API returned {success, data: {...}}, unwrap `data`
  const raw = apiResponse.data || apiResponse;

  // Ensure risk_percent always exists (some endpoints return risk_score 0-1)
  const riskScore = raw.risk_score ?? 0;
  const riskPercent = raw.risk_percent ?? riskScore * 100;

  return {
    ...raw,
    risk_score: riskScore,
    risk_percent: riskPercent,
    // Legacy aliases that some components referenced
    risk_percentage: riskPercent,
    authenticity_percentage: raw.authenticity_score ?? (100 - riskPercent),
  };
}

function extractError(err) {
  return err.response?.data?.error || err.response?.data?.detail || err.message || 'An error occurred';
}

const useForensicStore = create((set) => ({
  // System Status
  systemStatus: {
    loaded_models: [],
    missing_models: [],
    corefakenet_available: false,
    fusion_mlp_available: false,
    vit_available: false,
    device: 'cpu',
    total: 0,
  },
  isStatusLoading: true,
  statusError: null,

  // History
  history: [],
  historyTotal: 0,
  isHistoryLoading: false,
  historyError: null,

  // Per-page analysis state (persists across navigation)
  imageAnalysis: createAnalysisSlice(),
  videoAnalysis: createAnalysisSlice(),
  audioAnalysis: createAnalysisSlice(),
  multimodalAnalysis: createAnalysisSlice(),

  // Pending file from drag-drop on Dashboard
  pendingFile: null,
  setPendingFile: (file) => set({ pendingFile: file }),
  clearPendingFile: () => set({ pendingFile: null }),

  // --- Analysis Actions ---

  runImageAnalysis: async (file, mode) => {
    set({ imageAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeImage(file, mode);
      if (data.success) {
        set({ imageAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Image analysis complete', 'success');
      } else {
        set({ imageAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Image analysis failed', 'error');
      }
    } catch (err) {
      set({ imageAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    }
  },

  runVideoAnalysis: async (file, fps, aggregation) => {
    set({ videoAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeVideo(file, fps, aggregation);
      if (data.success) {
        set({ videoAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Video analysis complete', 'success');
      } else {
        set({ videoAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Video analysis failed', 'error');
      }
    } catch (err) {
      set({ videoAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    }
  },

  runAudioAnalysis: async (file) => {
    set({ audioAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeAudio(file);
      if (data.success) {
        set({ audioAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Audio analysis complete', 'success');
      } else {
        set({ audioAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Audio analysis failed', 'error');
      }
    } catch (err) {
      set({ audioAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    }
  },

  runMultimodalAnalysis: async (image, video, audio) => {
    set({ multimodalAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeMultimodal(image, video, audio);
      if (data.success) {
        set({ multimodalAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Multimodal analysis complete', 'success');
      } else {
        set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Multimodal analysis failed', 'error');
      }
    } catch (err) {
      set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    }
  },

  clearAnalysis: (type) => {
    set({ [`${type}Analysis`]: createAnalysisSlice() });
  },

  fetchStatus: async () => {
    set({ isStatusLoading: true, statusError: null });
    try {
      const data = await forensicApi.getStatus();
      const loaded = data.loaded || [];
      set({
        systemStatus: {
          loaded_models: loaded,
          missing_models: data.missing || [],
          corefakenet_available: data.corefakenet_ready || false,
          fusion_mlp_available: loaded.some(
            (m) => m.toLowerCase().includes('fusion') || m.toLowerCase().includes('mlp'),
          ),
          vit_available: loaded.some((m) => m.toLowerCase().includes('vit')),
          device: 'auto',
          total: data.total || 0,
        },
        isStatusLoading: false,
      });
    } catch (error) {
      set({ statusError: extractError(error), isStatusLoading: false });
    }
  },

  fetchHistory: async (limit = 20, mediaType = null) => {
    set({ isHistoryLoading: true, historyError: null });
    try {
      const data = await forensicApi.getHistory(limit, mediaType);
      set({
        history: data.data || [],
        historyTotal: data.total || 0,
        isHistoryLoading: false,
      });
    } catch (error) {
      set({ historyError: extractError(error), isHistoryLoading: false });
    }
  },
}));

export default useForensicStore;
