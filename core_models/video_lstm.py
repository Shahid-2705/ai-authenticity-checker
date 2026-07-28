"""
Video Temporal LSTM — learned replacement for the hand-tuned TemporalAnalyzer
heuristic in pipeline/video_analyzer.py.

The existing per-frame ensemble (ViT + Texture + Frequency + Face + DINOv2 +
EfficientNet Auth + Forensic -> Fusion MLP) already scores each frame
independently. TemporalAnalyzer then bumps the risk using fixed thresholds on
score variance and frame-to-frame jumps (see video_analyzer.py). That's a
reasonable heuristic but it's not learned from data — this module replaces it
with a small LSTM trained on real labeled video sequences to recognize
genuine temporal-inconsistency patterns (the same idea papers use: EVM+ResNext+LSTM
getting 97.76% on FF++ vs. frame-only baselines in the 90s).

Input per frame (already computed by ModelEnsemble.predict(), no new
per-frame model needed): [raw_risk, vit_prob, face_prob, frequency_prob].
Output: a single video-level P(fake) for the window.

Saves as: models/video_lstm.pth
"""

import torch
import torch.nn as nn


class VideoTemporalLSTM(nn.Module):
    """
    LSTM over a fixed-length window of per-frame signals.

    Architecture:
        LSTM(n_features, hidden_size) -> take final hidden state
        -> Linear(hidden_size, 16) -> ReLU -> Dropout -> Linear(16, 1) -> Sigmoid
    """

    FEATURE_NAMES = ["raw_risk", "vit_prob", "face_prob", "frequency_prob"]

    def __init__(self, n_features=4, hidden_size=32, window_size=10):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence, lengths=None):
        """
        Args:
            sequence: (batch, window_size, n_features) float tensor.
            lengths: optional (batch,) tensor of actual (unpadded) lengths,
                     for variable-length training batches.

        Returns:
            (batch, 1) video-level P(fake).
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(sequence)
        final_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(final_hidden)

    def predict_window(self, frame_results):
        """
        Convenience method for inference on a list of per-frame result dicts
        (as produced by ModelEnsemble.predict(), most recent frames last).

        Missing/short windows are zero-padded at the front, matching the
        causal, streaming behavior of the TemporalAnalyzer it replaces.

        Args:
            frame_results: list of dicts, each with at least
                           "raw_risk" (or "frame_risk" as fallback),
                           "vit_prob", "face_prob", "frequency_prob".

        Returns:
            float video-level P(fake) in [0, 1].
        """
        window = frame_results[-self.window_size:]
        feats = []
        for r in window:
            risk = r.get("raw_risk", r.get("frame_risk", 0.0))
            feats.append([
                risk,
                r.get("vit_prob", 0.0),
                r.get("face_prob", 0.0),
                r.get("frequency_prob", 0.0),
            ])
        while len(feats) < self.window_size:
            feats.insert(0, [0.0] * self.n_features)

        x = torch.tensor([feats], dtype=torch.float32)
        with torch.no_grad():
            return self.forward(x).item()
