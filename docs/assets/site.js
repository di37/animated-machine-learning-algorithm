// Gallery filters, the full-video player, and shareable links (#v=<video id>) to any video.
(() => {
  const dialog = document.getElementById("player");
  const video = dialog.querySelector("video");
  const title = dialog.querySelector("#player-title");
  const meta = dialog.querySelector(".player-meta");
  const openLink = dialog.querySelector("[data-open]");
  const copyButton = dialog.querySelector("[data-copy]");
  const cards = [...document.querySelectorAll(".card")];
  const groups = [...document.querySelectorAll(".group")];
  const filters = { dataset: "all", kind: "all" };

  function applyFilters() {
    groups.forEach((group) => {
      const datasetOk = filters.dataset === "all" || group.dataset.dataset === filters.dataset;
      const kindOk = filters.kind === "all" || group.dataset.kind === filters.kind;
      group.hidden = !(datasetOk && kindOk);
    });
  }

  document.querySelectorAll(".chip").forEach((chip) => {
    const attribute = "dataset" in chip.dataset ? "dataset" : "kind";
    chip.addEventListener("click", () => {
      filters[attribute] = chip.dataset[attribute];
      chip.parentElement.querySelectorAll(".chip").forEach((other) =>
        other.setAttribute("aria-pressed", String(other === chip)));
      applyFilters();
    });
  });

  function linkFor(card) {
    return `${location.origin}${location.pathname}#v=${card.dataset.id}`;
  }

  function play(card) {
    title.textContent = card.dataset.title;
    meta.textContent = card.dataset.meta;
    openLink.href = card.dataset.src;
    copyButton.textContent = "🔗 Copy link";
    copyButton.onclick = async () => {
      try {
        await navigator.clipboard.writeText(linkFor(card));
        copyButton.textContent = "✓ Link copied";
      } catch {
        copyButton.textContent = "Copy failed: use the address bar";
      }
    };
    video.src = card.dataset.src;
    history.replaceState(null, "", `#v=${card.dataset.id}`);
    dialog.showModal();
    video.play().catch(() => {}); // autoplay may be blocked; the controls still work
  }

  function close() {
    video.pause();
    video.removeAttribute("src");
    video.load();
    if (location.hash.startsWith("#v=")) history.replaceState(null, "", location.pathname);
  }

  cards.forEach((card) => card.addEventListener("click", () => play(card)));
  dialog.querySelector("[data-close]").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", close);
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

  const shared = location.hash.match(/^#v=([\w-]+)$/);
  if (shared) {
    const card = cards.find((c) => c.dataset.id === shared[1]);
    if (card) {
      card.scrollIntoView({ block: "center" });
      play(card);
    }
  }
})();
