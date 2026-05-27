(function () {
  const maxToasts = 4;

  function toast(title, body, options = {}) {
    const stack = document.getElementById("toast-stack");
    if (!stack) {
      return;
    }
    const item = document.createElement("div");
    item.className = `toast ${options.kind === "alert" ? "toast-alert" : "toast-device"}`;
    item.innerHTML = `<strong>${title}</strong><p>${body}</p>`;
    stack.prepend(item);
    while (stack.children.length > maxToasts) {
      stack.lastElementChild.remove();
    }
    window.setTimeout(() => {
      item.style.opacity = "0";
      item.style.transform = "translateY(-6px)";
      window.setTimeout(() => item.remove(), 250);
    }, options.timeout || 5200);
  }

  window.P169Notifications = { toast };
})();
