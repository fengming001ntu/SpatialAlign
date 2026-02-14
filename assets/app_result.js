// app_result.js

(() => {

const container = document.getElementById("container_r");
const prevBtn = document.getElementById("prevBtn_r");
const nextBtn = document.getElementById("nextBtn_r");
const pageInfo = document.getElementById("pageInfo_r");

// 3 行（每行 1 条样例）
const ROWS_PER_PAGE = 1;
const PAGE_SIZE = ROWS_PER_PAGE;

const METHOD_LABELS = ["Wan2.1-1.3B", "LTX-Video", "HunyuanVideo-1.5", "CogVideoX-1.5", "Ours"];

let DATA = [];
let page = 0;
let totalPages = 1;

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function highlightSpatial(text){
  if (!text) return "";
  const patterns = [
    "on the left of", "on the right of", "on the top of", "on the bottom of",
    "to the left of", "to the right of", "to the top of", "to the bottom of",
  ];
  let out = text;
  for (const ph of patterns){
    const re = new RegExp(`\\b${ph}\\b`, "gi");
    out = out.replace(re, m => `<strong class="rel">${m}</strong>`);
  }
  return out;
}

function splitFromPromptId(pid){
  if (!pid) return "eval120";
  if (pid.startsWith("test")) return "eval30";
  if (/^1\d{3}/.test(pid)) return "eval120";
  return "eval120";
}

function buildVideoPaths(filename){
  return [
    `assets/video_wan/${filename}`,
    `assets/video_ltx/${filename}`,
    `assets/video_hy15/${filename}`,
    `assets/video_cog/${filename}`,
    `assets/video_ours/${filename}`,
  ];
}

function curvePathFromVideoSrc(videoSrc){
  if (!videoSrc) return "";
  return videoSrc
    .replace("/video_", "/curve_")
    .replace(/\.mp4$/i, ".png");
}


function renderPage() {
  const start = page * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, DATA.length);
  const slice = DATA.slice(start, end);

  container.innerHTML = slice.map(d => {
    const prompt = highlightSpatial(escapeHtml(d.prompt || ""));
    const vids = buildVideoPaths(d.video, d.prompt_id);

    // vids[i] 里已经是 "video_xxx/xxx.mp4" 这种相对路径，直接用即可
    const colsHtml = METHOD_LABELS.map((label, i) => {
      const src = vids[i] || "";
      const curve = curvePathFromVideoSrc(src);
      return `
        <div class="video-col-r">
          <div class="video-label-r">${escapeHtml(label)}</div>
          <video class="video-r" autoplay muted loop playsinline preload="metadata"
                 src="${escapeHtml(src)}"></video>
          <img class="curve-r" src="${escapeHtml(curve)}"
                 alt="curve"
                 onerror="this.style.display='none'">
        </div>
      `;
    }).join("");

    return `
      <div class="item-r">
        <div class="prompt-r"><strong>Prompt:</strong> ${prompt}</div>
        <div class="videos-4">
          ${colsHtml}
        </div>
      </div>
    `;
  }).join("");

  totalPages = Math.max(1, Math.ceil(DATA.length / PAGE_SIZE));
  pageInfo.textContent = `Page ${page + 1} / ${totalPages}`;

  if (prevBtn) prevBtn.disabled = (page === 0);
  if (nextBtn) nextBtn.disabled = (page >= totalPages - 1);
}

if (prevBtn) {
  prevBtn.addEventListener("click", () => {
    if (page > 0) {
      page -= 1;
      renderPage();
    }
  });
}

if (nextBtn) {
  nextBtn.addEventListener("click", () => {
    if (page < totalPages - 1) {
      page += 1;
      renderPage();
    }
  });
}

(() => {
  const data = window.DATA_FILTERED;
  DATA = Array.isArray(data) ? data : [];
  page = 0;
  renderPage();
})();


})();