// Copy button for article code blocks. Adds a "copy" button to the top-right
// corner of each block that copies the code to the clipboard. Without
// JavaScript, no button is added and the code stays selectable as before.
(() => {
  // The async Clipboard API is missing on plain-http pages and can be refused
  // by browser policy; the legacy copy command still works inside a click.
  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const copied = document.execCommand("copy");
      area.remove();
      if (!copied) throw new Error("copy command refused");
    }
  };

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
        await copyText(code.textContent.replace(/\n$/, ""));
        button.textContent = "copied";
      } catch {
        button.textContent = "failed";
      }
      setTimeout(() => (button.textContent = "copy"), 2000);
    });
    box.append(button);
  }
})();
