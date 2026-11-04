// Copy button for article code blocks. Adds a "copy" button to the top-right
// corner of each block that copies the code to the clipboard. Without
// JavaScript, or without the async Clipboard API (plain-http pages), no button
// is added and the code stays selectable as before.
(() => {
  if (!navigator.clipboard) return;

  for (const pre of document.querySelectorAll("article pre")) {
    const code = pre.querySelector("code") || pre;
    // Highlighted blocks already sit in Chroma's div.highlight; bare ones get
    // a wrapper, so the button is positioned against the block, not the page.
    let box = pre.parentElement;
    if (!box.classList.contains("highlight")) {
      box = document.createElement("div");
      box.className = "code-block";
      pre.before(box);
      box.append(pre);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "copy";
    button.addEventListener("click", async () => {
      try {
        // Drop the final newline so a pasted command waits for Enter.
        await navigator.clipboard.writeText(code.textContent.replace(/\n$/, ""));
        button.textContent = "copied";
      } catch {
        button.textContent = "failed";
      }
      setTimeout(() => (button.textContent = "copy"), 2000);
    });
    box.append(button);
  }
})();
