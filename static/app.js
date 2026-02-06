(function () {
  "use strict";

  // --- Elements ---
  var btnCam = document.getElementById("btn-cam");
  var btnCapture = document.getElementById("btn-capture");
  var btnGen = document.getElementById("btn-generate");
  var btnDownload = document.getElementById("btn-download");
  var camSelect = document.getElementById("cam-select");
  var statusText = document.getElementById("status-text");
  var inputPrompt = document.getElementById("input-prompt");
  var infoDetect = document.getElementById("info-detect");
  var resultLabel = document.getElementById("result-label");
  var resultSection = document.getElementById("result-section");
  var progressContainer = document.getElementById("progress-container");
  var progressBar = document.getElementById("progress-bar");
  var progressText = document.getElementById("progress-text");

  var canvasDetect = document.getElementById("canvas-detect");
  var canvasMask = document.getElementById("canvas-mask");
  var canvasResult = document.getElementById("canvas-result");
  var ctxDetect = canvasDetect.getContext("2d");
  var ctxMask = canvasMask.getContext("2d");
  var ctxResult = canvasResult.getContext("2d");

  var video = document.getElementById("webcam-video");
  var captureCanvas = document.getElementById("capture-canvas");
  var captureCtx = captureCanvas.getContext("2d");

  // Status elements
  var chipYolo = document.getElementById("chip-yolo");
  var chipInpaint = document.getElementById("chip-inpaint");
  var modelBanner = document.getElementById("model-banner");
  var bannerDetail = document.getElementById("banner-detail");

  // --- State ---
  var stream = null;
  var wsDetect = null;
  var wsInpaint = null;
  var detecting = false;
  var waitingForResponse = false;
  var inpaintReady = false;
  var isGenerating = false;
  var frameCaptured = false;
  var hasGeneratedImage = false;

  // --- Button state management ---
  function updateGenerateButton() {
    if (isGenerating) return;
    if (!inpaintReady) {
      btnGen.disabled = true;
      btnGen.textContent = "Model Loading\u2026";
    } else if (!frameCaptured) {
      btnGen.disabled = true;
      btnGen.textContent = "Generate";
    } else {
      btnGen.disabled = false;
      btnGen.textContent = "Generate";
    }
  }

  function updateCaptureButton() {
    btnCapture.disabled = !detecting;
  }

  function showResultSection() {
    resultSection.hidden = false;
  }

  // --- Model status polling ---
  function pollModelStatus() {
    fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        chipYolo.className = "status-chip " + (data.yolo === "ready" ? "ready" : "loading");

        if (data.inpaint === "ready") {
          chipInpaint.className = "status-chip ready";
          modelBanner.classList.add("hidden");
          inpaintReady = true;
          updateGenerateButton();
        } else if (data.inpaint === "error") {
          chipInpaint.className = "status-chip error";
          modelBanner.classList.add("hidden");
          inpaintReady = false;
          updateGenerateButton();
        } else {
          chipInpaint.className = "status-chip loading";
          bannerDetail.textContent = data.inpaint_detail || "Loading...";
          inpaintReady = false;
          updateGenerateButton();
        }

        if (data.inpaint !== "ready" && data.inpaint !== "error") {
          setTimeout(pollModelStatus, 3000);
        }
      })
      .catch(function () {
        setTimeout(pollModelStatus, 5000);
      });
  }

  pollModelStatus();

  // --- Enumerate cameras ---
  function populateCameras() {
    navigator.mediaDevices.enumerateDevices().then(function (devices) {
      while (camSelect.options.length > 1) camSelect.remove(1);
      devices.forEach(function (d) {
        if (d.kind === "videoinput") {
          var opt = document.createElement("option");
          opt.value = d.deviceId;
          opt.textContent = d.label || ("Camera " + camSelect.options.length);
          camSelect.appendChild(opt);
        }
      });
    });
  }

  navigator.mediaDevices.getUserMedia({ video: true }).then(function (tmp) {
    tmp.getTracks().forEach(function (t) { t.stop(); });
    populateCameras();
  }).catch(function () {
    populateCameras();
  });

  // --- WebSocket helpers ---
  function wsUrl(path) {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + path;
  }

  function connectDetectWs() {
    wsDetect = new WebSocket(wsUrl("/ws/detect"));
    wsDetect.binaryType = "arraybuffer";

    wsDetect.onopen = function () {
      statusText.textContent = "Connected \u2014 streaming";
      detecting = true;
      updateCaptureButton();
      sendFrame();
    };

    wsDetect.onmessage = function (ev) {
      var resp = JSON.parse(ev.data);
      drawBase64(ctxDetect, canvasDetect, resp.detect);
      drawBase64(ctxMask, canvasMask, resp.mask);
      infoDetect.textContent = resp.faces + " face" + (resp.faces !== 1 ? "s" : "") + " detected";

      waitingForResponse = false;
      if (detecting) {
        sendFrame();
      }
    };

    wsDetect.onclose = function () {
      detecting = false;
      updateCaptureButton();
      statusText.textContent = "Detect WS closed";
    };

    wsDetect.onerror = function () {
      detecting = false;
      updateCaptureButton();
      statusText.textContent = "Detect WS error";
    };
  }

  function connectInpaintWs() {
    wsInpaint = new WebSocket(wsUrl("/ws/inpaint"));

    wsInpaint.onmessage = function (ev) {
      var msg = JSON.parse(ev.data);

      if (msg.error) {
        alert("Error: " + msg.error);
        isGenerating = false;
        updateGenerateButton();
        progressContainer.hidden = true;
        return;
      }

      if (msg.status === "started") {
        resultLabel.textContent = "GENERATING\u2026";
        showResultSection();
        progressContainer.hidden = false;
        progressBar.style.width = "0%";
        progressText.textContent = "Starting\u2026";
      } else if (msg.status === "progress") {
        var pct = Math.round((msg.step / msg.total_steps) * 100);
        progressBar.style.width = pct + "%";
        progressText.textContent = "Step " + msg.step + "/" + msg.total_steps + " (" + msg.elapsed + "s)";
      } else if (msg.status === "done") {
        progressBar.style.width = "100%";
        progressText.textContent = "Done in " + msg.elapsed + "s";
        drawBase64(ctxResult, canvasResult, msg.image);
        resultLabel.textContent = "GENERATED";
        isGenerating = false;
        hasGeneratedImage = true;
        btnDownload.disabled = false;
        updateGenerateButton();
        setTimeout(function () { progressContainer.hidden = true; }, 3000);
      }
    };

    wsInpaint.onclose = function () {
      isGenerating = false;
      updateGenerateButton();
    };
  }

  // --- Frame capture & send ---
  function sendFrame() {
    if (!detecting || !wsDetect || wsDetect.readyState !== WebSocket.OPEN) return;
    if (waitingForResponse) return;

    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 480;
    captureCtx.drawImage(video, 0, 0);

    captureCanvas.toBlob(
      function (blob) {
        if (blob && wsDetect && wsDetect.readyState === WebSocket.OPEN) {
          waitingForResponse = true;
          blob.arrayBuffer().then(function (buf) {
            wsDetect.send(buf);
          });
        }
      },
      "image/jpeg",
      0.75
    );
  }

  // --- Draw base64 image to canvas ---
  function drawBase64(ctx, canvas, b64) {
    var img = new Image();
    img.onload = function () {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = "data:image/jpeg;base64," + b64;
  }

  // --- Camera toggle ---
  btnCam.addEventListener("click", function () {
    if (stream) {
      detecting = false;
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
      if (wsDetect) wsDetect.close();
      wsDetect = null;
      btnCam.textContent = "Start Webcam";
      statusText.textContent = "Webcam off";
      frameCaptured = false;
      updateCaptureButton();
      updateGenerateButton();
    } else {
      var deviceId = camSelect.value;
      var constraints = { video: { width: 640, height: 480 } };
      if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
      }
      btnCam.disabled = true;
      statusText.textContent = "Requesting camera\u2026";
      navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (s) {
          stream = s;
          video.srcObject = s;
          btnCam.textContent = "Stop Webcam";
          btnCam.disabled = false;
          connectDetectWs();
        })
        .catch(function (err) {
          statusText.textContent = "Camera error: " + err.message;
          btnCam.disabled = false;
        });
    }
  });

  // --- Capture frame ---
  btnCapture.addEventListener("click", function () {
    btnCapture.disabled = true;
    btnCapture.textContent = "Capturing\u2026";

    fetch("/api/capture", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          alert("Capture failed: " + data.error);
          btnCapture.disabled = false;
          btnCapture.textContent = "Capture Frame";
          return;
        }

        // Show the captured mask preview in the result panel
        showResultSection();
        drawBase64(ctxResult, canvasResult, data.mask);
        resultLabel.textContent = "CAPTURED PREVIEW";
        hasGeneratedImage = false;
        btnDownload.disabled = true;

        frameCaptured = true;
        btnCapture.textContent = "Recapture";
        btnCapture.disabled = false;
        updateGenerateButton();
      })
      .catch(function () {
        alert("Capture request failed.");
        btnCapture.disabled = false;
        btnCapture.textContent = "Capture Frame";
      });
  });

  // --- Generate ---
  btnGen.addEventListener("click", function () {
    var prompt = inputPrompt.value.trim();
    if (!prompt) {
      alert("Enter a prompt first.");
      return;
    }

    isGenerating = true;
    btnGen.disabled = true;
    btnGen.textContent = "Generating\u2026";

    if (!wsInpaint || wsInpaint.readyState !== WebSocket.OPEN) {
      connectInpaintWs();
      wsInpaint.onopen = function () {
        wsInpaint.send(JSON.stringify({ prompt: prompt }));
      };
    } else {
      wsInpaint.send(JSON.stringify({ prompt: prompt }));
    }
  });

  // --- Download generated image ---
  btnDownload.addEventListener("click", function () {
    if (!hasGeneratedImage) return;
    var link = document.createElement("a");
    link.download = "inpainted-" + Date.now() + ".png";
    link.href = canvasResult.toDataURL("image/png");
    link.click();
  });
})();
