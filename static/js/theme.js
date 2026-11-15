(() => {
  const root = document.documentElement;
  const storageKey = "nakul-theme";
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  const validTheme = (value) => value === "light" || value === "dark";
  let preference;

  try {
    const saved = localStorage.getItem(storageKey);
    if (validTheme(saved)) preference = saved;
  } catch {
    // The switch still works when browser storage is unavailable.
  }

  function applyTheme() {
    root.dataset.theme = preference || (systemTheme.matches ? "dark" : "light");
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    button.querySelector("[data-theme-label]").textContent = next === "light" ? "Light" : "Dark";
    button.setAttribute("aria-label", `Switch to ${next} mode`);
    button.title = `Switch to ${next} mode`;
    button.hidden = false;
  }

  // Run before styles load so a saved choice is applied before the first paint.
  applyTheme();
  document.addEventListener("DOMContentLoaded", () => {
    applyTheme();
    document.getElementById("theme-toggle").addEventListener("click", () => {
      preference = root.dataset.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(storageKey, preference);
      } catch {
        // Keep the in-page choice even if it cannot be persisted.
      }
      applyTheme();
    });
  });

  systemTheme.addEventListener("change", () => {
    if (!preference) applyTheme();
  });
  window.addEventListener("storage", (event) => {
    if (event.key !== storageKey && event.key !== null) return;
    preference = validTheme(event.newValue) ? event.newValue : undefined;
    applyTheme();
  });
})();
