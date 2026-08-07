import { create } from 'zustand';

const MAX_TOASTS = 5;
let nextId = 0;
const timers = new Map();

const useToastStore = create((set) => ({
  toasts: [],

  addToast: (message, type = 'info', duration = 4000) => {
    const id = ++nextId;

    set((state) => {
      // Evict oldest toasts beyond the cap
      const incoming = [...state.toasts, { id, message, type }];
      const evicted = incoming.length > MAX_TOASTS ? incoming.slice(incoming.length - MAX_TOASTS) : incoming;

      // Clear timers for evicted toasts
      for (const t of state.toasts) {
        if (!evicted.find((e) => e.id === t.id) && timers.has(t.id)) {
          clearTimeout(timers.get(t.id));
          timers.delete(t.id);
        }
      }

      return { toasts: evicted };
    });

    if (duration > 0) {
      const timer = setTimeout(() => {
        timers.delete(id);
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
      }, duration);
      timers.set(id, timer);
    }
  },

  removeToast: (id) => {
    if (timers.has(id)) {
      clearTimeout(timers.get(id));
      timers.delete(id);
    }
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
  },
}));

export default useToastStore;
