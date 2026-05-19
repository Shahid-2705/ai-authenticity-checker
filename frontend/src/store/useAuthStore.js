import { create } from 'zustand';
import { supabase, isAuthEnabled } from '../services/supabase';

// Holds the Supabase auth subscription so we can clean it up
// before registering a new one (e.g. React StrictMode double-mount).
let authSubscription = null;

const useAuthStore = create((set) => ({
  user: null,
  session: null,
  isLoading: true,

  initialize: async () => {
    if (!isAuthEnabled()) {
      set({ isLoading: false });
      return;
    }

    // Clean up any previous listener before registering a new one
    if (authSubscription) {
      authSubscription.unsubscribe();
      authSubscription = null;
    }

    const { data: { session } } = await supabase.auth.getSession();
    set({
      session,
      user: session?.user ?? null,
      isLoading: false,
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      set({
        session,
        user: session?.user ?? null,
      });
    });
    authSubscription = subscription;
  },

  signIn: async (email, password) => {
    if (!isAuthEnabled()) return { error: { message: 'Auth not configured' } };
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (!error) {
      set({ session: data.session, user: data.user });
    }
    return { data, error };
  },

  signUp: async (email, password) => {
    if (!isAuthEnabled()) return { error: { message: 'Auth not configured' } };
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (!error && data.session) {
      set({ session: data.session, user: data.user });
    }
    return { data, error };
  },

  resetPassword: async (email) => {
    if (!isAuthEnabled()) return { error: { message: 'Auth not configured' } };
    const { error } = await supabase.auth.resetPasswordForEmail(email);
    return { error };
  },

  signOut: async () => {
    if (!isAuthEnabled()) return;
    await supabase.auth.signOut();
    set({ session: null, user: null });
  },
}));

export default useAuthStore;
