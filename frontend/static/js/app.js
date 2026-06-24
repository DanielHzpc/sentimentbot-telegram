const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const reloadBtn = document.getElementById("reloadBtn");
const errorBox = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");
const sentimentValue = document.getElementById("sentimentValue");
const confidenceValue = document.getElementById("confidenceValue");
const explanationValue = document.getElementById("explanationValue");
const historyList = document.getElementById("historyList");
const peopleResults = document.getElementById("peopleResults");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const liveStatus = document.getElementById("liveStatus");

let cameraStream = null;
let detectionTimer = null;
let detectionInProgress = false;
let lastDetectedFaces = [];
let nativeFaceDetector = null;
let nativeDetectorAvailable = false;
let liveDetectionMode = "Servidor OpenCV";

const FACE_COLORS = ["#ef4444", "#2563eb"];
const FACE_LABELS = ["Persona 1", "Persona 2"];


function setupNativeFaceDetector() {
    if ("FaceDetector" in window) {
        try {
            nativeFaceDetector = new FaceDetector({ fastMode: true, maxDetectedFaces: 2 });
            nativeDetectorAvailable = true;
            liveDetectionMode = "Detector nativo del navegador";
        } catch (error) {
            nativeFaceDetector = null;
            nativeDetectorAvailable = false;
            liveDetectionMode = "Servidor OpenCV";
        }
    }
}

function normalizeBrowserFaces(detectedFaces = []) {
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    return detectedFaces
        .map((face) => {
            const box = face.boundingBox || face;
            return {
                x: Math.max(0, Math.round(box.x ?? box.left ?? 0)),
                y: Math.max(0, Math.round(box.y ?? box.top ?? 0)),
                w: Math.min(width, Math.round(box.width ?? box.w ?? 0)),
                h: Math.min(height, Math.round(box.height ?? box.h ?? 0)),
            };
        })
        .filter((face) => face.w > 30 && face.h > 30)
        .sort((a, b) => (b.w * b.h) - (a.w * a.h))
        .slice(0, 2);
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function getVideoImageBase64(quality = 0.75) {
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);

    return canvas.toDataURL("image/jpeg", quality);
}

function resizeOverlay() {
    const width = video.videoWidth || video.clientWidth || 640;
    const height = video.videoHeight || video.clientHeight || 480;

    overlayCanvas.width = width;
    overlayCanvas.height = height;
}

function clearOverlay() {
    const context = overlayCanvas.getContext("2d");
    context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function drawFaceBoxes(faces = []) {
    resizeOverlay();
    clearOverlay();

    const context = overlayCanvas.getContext("2d");
    context.lineWidth = Math.max(3, Math.round(overlayCanvas.width / 220));
    context.font = `${Math.max(16, Math.round(overlayCanvas.width / 34))}px Arial`;
    context.textBaseline = "top";

    faces.slice(0, 2).forEach((face, index) => {
        const color = FACE_COLORS[index] || "#22c55e";
        const label = FACE_LABELS[index] || `Persona ${index + 1}`;

        context.strokeStyle = color;
        context.fillStyle = color;
        context.strokeRect(face.x, face.y, face.w, face.h);

        const textWidth = context.measureText(label).width + 18;
        const labelHeight = 28;
        const labelY = Math.max(0, face.y - labelHeight);

        context.fillRect(face.x, labelY, textWidth, labelHeight);
        context.fillStyle = "#ffffff";
        context.fillText(label, face.x + 9, labelY + 5);
    });
}

function updateLiveStatus(count) {
    if (!cameraStream) {
        liveStatus.textContent = "Cámara apagada.";
        return;
    }

    const modeText = nativeDetectorAvailable
        ? "Detector nativo del navegador activo"
        : "Detector OpenCV del servidor activo";

    if (count === 0) {
        liveStatus.textContent = `${modeText}. No hay rostros detectados todavía.`;
    } else if (count === 1) {
        liveStatus.textContent = `${modeText}. Detectando 1 rostro: Persona 1 en rojo.`;
    } else {
        liveStatus.textContent = `${modeText}. Detectando 2 rostros: Persona 1 rojo, Persona 2 azul.`;
    }
}

async function detectFacesLive() {
    if (!cameraStream || detectionInProgress || video.readyState < 2) return;

    detectionInProgress = true;

    try {
        // 1) Mejor opción: detector nativo del navegador. En Chrome/Edge suele detectar
        // rostros más estable que Haar Cascade de OpenCV, especialmente con cambios de luz.
        if (nativeFaceDetector) {
            const browserFaces = await nativeFaceDetector.detect(video);
            lastDetectedFaces = normalizeBrowserFaces(browserFaces);
            drawFaceBoxes(lastDetectedFaces);
            updateLiveStatus(lastDetectedFaces.length);
            return;
        }

        // 2) Respaldo: detección en backend con OpenCV.
        const imageBase64 = getVideoImageBase64(0.55);
        const response = await fetch("/api/detectar-rostros", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ imagen: imageBase64, rostros: lastDetectedFaces }),
        });

        const data = await response.json();
        if (response.ok && data.ok) {
            lastDetectedFaces = data.rostros || [];
            drawFaceBoxes(lastDetectedFaces);
            updateLiveStatus(lastDetectedFaces.length);
        }
    } catch (error) {
        // Si falla el detector nativo una vez, pasamos al respaldo OpenCV.
        if (nativeFaceDetector) {
            nativeFaceDetector = null;
            nativeDetectorAvailable = false;
            liveDetectionMode = "Servidor OpenCV";
        }
    } finally {
        detectionInProgress = false;
    }
}

function startLiveDetection() {
    stopLiveDetection();
    detectionTimer = setInterval(detectFacesLive, nativeDetectorAvailable ? 220 : 650);
    detectFacesLive();
}

function stopLiveDetection() {
    if (detectionTimer) {
        clearInterval(detectionTimer);
        detectionTimer = null;
    }
    detectionInProgress = false;
    lastDetectedFaces = [];
    clearOverlay();
}

function showResult(data) {
    sentimentValue.textContent = data.sentimiento || "-";
    confidenceValue.textContent = `${data.confianza ?? 0}%`;
    explanationValue.textContent = `${data.explicacion || ""}\nModo: ${data.modo_analisis || "IA/local"}`;

    const results = data.resultados || [];
    if (results.length > 0) {
        peopleResults.innerHTML = results.map((item, index) => {
            const personClass = index === 0 ? "person-red" : "person-blue";
            const faceImage = item.face_image
                ? `<img class="face-snapshot" src="${item.face_image}" alt="Foto del rostro de ${escapeHtml(item.persona || `Persona ${index + 1}`)}">`
                : `<div class="face-snapshot face-placeholder">Sin foto</div>`;

            return `
                <article class="person-result ${personClass}">
                    ${faceImage}
                    <div class="person-result-content">
                        <div class="person-title">
                            <span>${escapeHtml(item.persona || `Persona ${index + 1}`)}</span>
                            <strong>${escapeHtml(item.sentimiento || "-")}</strong>
                        </div>
                        <p><b>Confianza:</b> ${item.confianza ?? 0}%</p>
                        <p>${escapeHtml(item.explicacion || "")}</p>
                    </div>
                </article>
            `;
        }).join("");
    } else {
        peopleResults.innerHTML = "";
    }

    if (results.length > 0) {
        drawFaceBoxes(results.map((item) => item.box).filter(Boolean));
    }

    resultCard.classList.remove("hidden");
}

async function startCamera() {
    hideError();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Tu navegador no permite activar la cámara desde esta página.");
        return;
    }

    try {
        setupNativeFaceDetector();

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
            },
            audio: false,
        });

        video.srcObject = cameraStream;
        video.classList.remove("hidden");
        overlayCanvas.classList.remove("hidden");
        cameraPlaceholder.classList.add("hidden");
        captureBtn.disabled = false;
        stopCameraBtn.disabled = false;
        startCameraBtn.disabled = true;
        liveStatus.textContent = "Cámara activa. Buscando rostros...";

        video.onloadedmetadata = () => {
            resizeOverlay();
            startLiveDetection();
        };
    } catch (error) {
        showError("No se pudo activar la cámara. Revisa permisos del navegador o prueba en Chrome/Edge.");
    }
}

function stopCamera() {
    stopLiveDetection();

    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    video.classList.add("hidden");
    overlayCanvas.classList.add("hidden");
    cameraPlaceholder.classList.remove("hidden");
    captureBtn.disabled = true;
    stopCameraBtn.disabled = true;
    startCameraBtn.disabled = false;
    liveStatus.textContent = "Cámara apagada.";
}

async function captureAndAnalyzeFace() {
    hideError();

    if (!cameraStream) {
        showError("Primero activa la cámara.");
        return;
    }

    const imageBase64 = getVideoImageBase64(0.9);

    captureBtn.disabled = true;
    captureBtn.textContent = "Analizando rostros...";

    try {
        const response = await fetch("/api/analizar-rostro", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ imagen: imageBase64, rostros: lastDetectedFaces }),
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
            showError(data.mensaje || "No se pudo analizar la imagen.");
            return;
        }

        if (!data.rostro_detectado) {
            showError(data.explicacion || "No se detectaron rostros.");
        }

        showResult(data);
        await loadHistory();
    } catch (error) {
        showError("Ocurrió un error conectando con el servidor local.");
    } finally {
        captureBtn.disabled = false;
        captureBtn.textContent = "Capturar y analizar";
    }
}

async function analyzeText() {
    hideError();

    const texto = textInput.value.trim();

    if (texto.length < 3) {
        showError("Escribe un texto de al menos 3 caracteres.");
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analizando...";

    try {
        const response = await fetch("/api/analizar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texto }),
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
            showError(data.mensaje || "No se pudo analizar el texto.");
            return;
        }

        showResult(data);
        textInput.value = "";
        await loadHistory();
    } catch (error) {
        showError("Ocurrió un error conectando con el servidor local.");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analizar texto";
    }
}

async function loadHistory() {
    const response = await fetch("/api/historial");
    const data = await response.json();

    if (!data.historial || data.historial.length === 0) {
        historyList.innerHTML = '<p class="empty">Todavía no hay análisis guardados.</p>';
        return;
    }

    historyList.innerHTML = data.historial.map((item) => `
        <article class="history-item">
            <p><strong>${escapeHtml(item.sentimiento)}</strong> · ${item.confianza}%</p>
            <p>"${escapeHtml(item.texto)}"</p>
            <p class="history-meta">${escapeHtml(item.explicacion)}<br>${escapeHtml(item.fecha_analisis)}</p>
        </article>
    `).join("");
}

async function clearHistory() {
    await fetch("/api/historial", { method: "DELETE" });
    await loadHistory();
    resultCard.classList.add("hidden");
}

startCameraBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureAndAnalyzeFace);
stopCameraBtn.addEventListener("click", stopCamera);
analyzeBtn.addEventListener("click", analyzeText);
clearBtn.addEventListener("click", clearHistory);
reloadBtn.addEventListener("click", loadHistory);
window.addEventListener("resize", () => drawFaceBoxes(lastDetectedFaces));

textInput.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        analyzeText();
    }
});

loadHistory();
