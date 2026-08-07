/**
 * Shared formatting utilities.
 */

/** Relative time string (e.g. "3m ago", "2h ago"). */
export function formatRelativeTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

/** Short date-time (e.g. "May 19, 14:32"). */
export function formatShortDateTime(timestamp) {
  if (!timestamp) return '';
  try {
    return new Date(timestamp).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(timestamp);
  }
}

/** File size in human-readable form. */
export function formatFileSize(bytes) {
  if (!bytes) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Accepted file extensions per media type. */
export const ACCEPTED_EXTENSIONS = {
  image: ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'],
  video: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
  audio: ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
};

/** Check if a file matches the expected accept type. */
export function isFileAccepted(file, accept) {
  if (!accept || accept === '*') return true;
  const type = file.type || '';
  const name = (file.name || '').toLowerCase();

  // Handle "image/*", "video/*", "audio/*"
  if (accept.endsWith('/*')) {
    const category = accept.replace('/*', '');
    if (type.startsWith(category + '/')) return true;
    const exts = ACCEPTED_EXTENSIONS[category] || [];
    return exts.some((ext) => name.endsWith(ext));
  }

  return type === accept || accept.split(',').some((a) => type === a.trim());
}

/** Map a file's MIME type to the appropriate analysis route. */
export function detectMediaRoute(file) {
  const t = file.type || '';
  if (t.startsWith('image/')) return '/image';
  if (t.startsWith('video/')) return '/video';
  if (t.startsWith('audio/')) return '/audio';
  return '/multimodal';
}
