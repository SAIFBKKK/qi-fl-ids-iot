(function () {
  const storageKey = "p16_9_dashboard_theme";
  const root = document.documentElement;

  function applyTheme(theme) {
    const normalized = theme === "dark" ? "dark" : "light";
    root.setAttribute("data-theme", normalized);
    localStorage.setItem(storageKey, normalized);
    const button = document.getElementById("theme-toggle");
    if (button) {
      button.textContent = normalized === "dark" ? "Light mode" : "Dark mode";
    }
  }

  window.P169Theme = { applyTheme };

  document.addEventListener("DOMContentLoaded", () => {
    const saved = localStorage.getItem(storageKey) || "light";
    applyTheme(saved);
    const button = document.getElementById("theme-toggle");
    if (button) {
      button.addEventListener("click", () => {
        applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
      });
    }
  });
})();
