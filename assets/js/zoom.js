// Click-to-zoom for article figures. Each figure links to its full-size image
// (the figure shortcode's link=); this opens that image in a <dialog> overlay on
// the same page instead of navigating away. Without JavaScript, or in a browser
// without <dialog>, the link still opens the image as before.
(() => {
  const links = [...document.querySelectorAll("article figure > a[href]")]
    .filter((a) => a.querySelector("img"));
  if (!links.length || typeof HTMLDialogElement !== "function") return;

  const dialog = document.createElement("dialog");
  dialog.className = "zoom";
  dialog.setAttribute("aria-label", "Enlarged figure");
  const img = document.createElement("img");
  dialog.append(img);
  document.body.append(dialog);

  for (const a of links) {
    a.addEventListener("click", (e) => {
      // Leave modified clicks alone so "open in new tab" still works.
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      img.src = a.href;
      img.alt = a.querySelector("img").alt;
      dialog.showModal();
    });
  }

  // A click anywhere (image or backdrop) closes it; Esc closes it natively.
  dialog.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => img.removeAttribute("src"));
})();
