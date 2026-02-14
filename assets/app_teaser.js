const container = document.getElementById("container");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const pageInfo = document.getElementById("pageInfo");

const COLS = 1;
const ROWS_PER_PAGE = 2;
const PAGE_SIZE = COLS * ROWS_PER_PAGE; // 6

let DATA = [];
let page = 0;
let totalPages = 1;

function highlightSpatial(text){
  if (!text) return "";

  const patterns = [
    "on the left of",
    "on the right of",
    "on the top of",
    "to the left of",
    "to the right of",
    "to the top of",
  ];

  let out = text;
  for (const ph of patterns){
    const re = new RegExp(`\\b${ph}\\b`, "gi");
    out = out.replace(re, m => `<strong class="rel">${m}</strong>`);
  }
  return out;
}


function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* ===== 路径规则（与你当前目录一致） ===== */
function videoPath(split, filename) {
  return `assets/${split}/${filename}`;
}

function curvePath(split, filename) {
  // curve 文件名是 xxx.mp4.png
  return `assets/${split}/${filename}.png`;
}

/* ===== 整体缩放（中轴线） ===== */
function autoScale() {
  const root = document.getElementById("scale-root");
  if (!root) return;   // ✅ 没有 scale-root 就什么都不做
  
  const baseWidth =
    parseInt(
      getComputedStyle(document.documentElement)
        .getPropertyValue("--base-width"),
      10
    ) || 1440;

  const scale = Math.min(window.innerWidth / baseWidth, 1);
  root.style.transform = `translateX(-50%) scale(${scale})`;
}

window.addEventListener("resize", autoScale);
window.addEventListener("load", autoScale);

/* ===== 渲染单页（只更新 container，不刷新页面） ===== */
function renderPage() {
  const start = page * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, DATA.length);
  const slice = DATA.slice(start, end);

  container.innerHTML = slice.map(d => {
    const v = d.video;
    const p = d.prompt;

    return `
      <div class="item">
        <div class="prompt">
          <strong>Prompt:</strong> ${highlightSpatial(escapeHtml(p || ""))}
        </div>

        <div class="videos">
          <div class="video-col">
            <div class="video-label">Wan2.1-1.3B</div>
            <video autoplay muted loop playsinline preload="metadata"
              src="${videoPath("video_wan", v)}"></video>

            
          </div>

          <div class="video-col">
            <div class="video-label">Ours</div>
            <video autoplay muted loop playsinline preload="metadata"
              src="${videoPath("video_ours", v)}"></video>

          </div>
        </div>
      </div>
    `;
  }).join("");

  pageInfo.textContent = `Page ${page + 1} / ${totalPages}`;
  prevBtn.disabled = (page === 0);
  nextBtn.disabled = (page >= totalPages - 1);

  autoScale();
}

prevBtn.addEventListener("click", () => {
  if (page > 0) {
    page -= 1;
    renderPage();
    window.scrollTo({ top: 0, behavior: "instant" });
  }
});

nextBtn.addEventListener("click", () => {
  if (page < totalPages - 1) {
    page += 1;
    renderPage();
    window.scrollTo({ top: 0, behavior: "instant" });
  }
});

/* ===== 首次加载数据（只加载一次） ===== */
// fetch("./data_filtered.json", { cache: "no-store" })
//   .then(r => {
//     if (!r.ok) throw new Error(`HTTP ${r.status} when loading data_filtered.json`);
//     return r.json();
//   })
//   .then(data => {
//     DATA = Array.isArray(data) ? data : [];
//     totalPages = Math.max(1, Math.ceil(DATA.length / PAGE_SIZE));
//     page = 0;
//     renderPage();
//   })
//   .catch(err => {
//     container.innerHTML = "<p><strong>Failed to load data_filtered.json</strong></p>";
//     console.error(err);
//   });

(() => {
  const data = window.DATA_FILTERED;
  DATA = Array.isArray(data) ? data.slice(0, 2) : []; // 只取前2条
  totalPages = Math.max(1, Math.ceil(DATA.length / PAGE_SIZE));
  page = 0;
  renderPage();
})();