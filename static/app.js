(function () {
  "use strict";

  // =========================================================================
  // DOM Elements
  // =========================================================================
  var el = {
    btnCam: document.getElementById("btn-cam"),
    btnCapture: document.getElementById("btn-capture"),
    btnGen: document.getElementById("btn-generate"),
    btnDownload: document.getElementById("btn-download"),
    camSelect: document.getElementById("cam-select"),
    modeSelect: document.getElementById("mode-select"),
    classSelect: document.getElementById("class-select"),
    statusText: document.getElementById("status-text"),
    headerSub: document.getElementById("header-sub"),
    inputPrompt: document.getElementById("input-prompt"),
    infoDetect: document.getElementById("info-detect"),
    resultLabel: document.getElementById("result-label"),
    resultSection: document.getElementById("result-section"),
    progressContainer: document.getElementById("progress-container"),
    progressBar: document.getElementById("progress-bar"),
    progressText: document.getElementById("progress-text"),
    canvasDetect: document.getElementById("canvas-detect"),
    canvasMask: document.getElementById("canvas-mask"),
    canvasResult: document.getElementById("canvas-result"),
    video: document.getElementById("webcam-video"),
    captureCanvas: document.getElementById("capture-canvas"),
    chipDetect: document.getElementById("chip-detect"),
    chipInpaint: document.getElementById("chip-inpaint"),
    modelBanner: document.getElementById("model-banner"),
    bannerDetail: document.getElementById("banner-detail")
  };

  var ctx = {
    detect: el.canvasDetect.getContext("2d"),
    mask: el.canvasMask.getContext("2d"),
    result: el.canvasResult.getContext("2d"),
    capture: el.captureCanvas.getContext("2d")
  };

  // =========================================================================
  // Application State
  // =========================================================================
  var state = {
    camera: {
      stream: null
    },
    detection: {
      ws: null,
      active: false,
      waitingForResponse: false
    },
    inpainting: {
      ws: null,
      ready: false,
      generating: false
    },
    ui: {
      frameCaptured: false,
      hasGeneratedImage: false
    }
  };

  var SUBTITLES = {
    face: "Face Segmentation \u00b7 Stable Diffusion",
    object: "Object Segmentation \u00b7 Stable Diffusion"
  };

  // =========================================================================
  // Utilities
  // =========================================================================
  var Util = {
    wsUrl: function (path) {
      var proto = location.protocol === "https:" ? "wss:" : "ws:";
      return proto + "//" + location.host + path;
    },

    drawBase64: function (targetCtx, canvas, b64) {
      var img = new Image();
      img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        targetCtx.drawImage(img, 0, 0);
      };
      img.src = "data:image/jpeg;base64," + b64;
    }
  };

  // =========================================================================
  // UI — button states, mode switching, progress display
  // =========================================================================
  var UI = {
    updateGenerateButton: function () {
      el.btnGen.classList.remove("btn-cancel");
      if (state.inpainting.generating) return;
      if (!state.inpainting.ready) {
        el.btnGen.disabled = true;
        el.btnGen.textContent = "Model Loading\u2026";
      } else if (!state.ui.frameCaptured) {
        el.btnGen.disabled = true;
        el.btnGen.textContent = "Generate";
      } else {
        el.btnGen.disabled = false;
        el.btnGen.textContent = "Generate";
      }
    },

    updateCaptureButton: function () {
      el.btnCapture.disabled = !state.detection.active;
    },

    showResultSection: function () {
      el.resultSection.hidden = false;
    },

    currentMode: function () {
      return el.modeSelect.value;
    },

    currentClasses: function () {
      return el.classSelect.value;
    },

    updateModeUI: function () {
      var mode = UI.currentMode();
      el.classSelect.style.display = mode === "object" ? "" : "none";
      el.headerSub.textContent = SUBTITLES[mode] || SUBTITLES.face;
    }
  };

  // =========================================================================
  // Camera — enumerate devices, start/stop webcam stream
  // =========================================================================
  var Camera = {
    populateDevices: function () {
      navigator.mediaDevices.enumerateDevices().then(function (devices) {
        while (el.camSelect.options.length > 1) el.camSelect.remove(1);
        devices.forEach(function (d) {
          if (d.kind === "videoinput") {
            var opt = document.createElement("option");
            opt.value = d.deviceId;
            opt.textContent = d.label || ("Camera " + el.camSelect.options.length);
            el.camSelect.appendChild(opt);
          }
        });
      });
    },

    start: function () {
      var deviceId = el.camSelect.value;
      var constraints = { video: { width: 640, height: 480 } };
      if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
      }
      el.btnCam.disabled = true;
      el.statusText.textContent = "Requesting camera\u2026";
      navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (s) {
          state.camera.stream = s;
          el.video.srcObject = s;
          el.btnCam.textContent = "Stop Webcam";
          el.btnCam.disabled = false;
          Detection.connect();
        })
        .catch(function (err) {
          el.statusText.textContent = "Camera error: " + err.message;
          el.btnCam.disabled = false;
        });
    },

    stop: function () {
      state.detection.active = false;
      state.camera.stream.getTracks().forEach(function (t) { t.stop(); });
      state.camera.stream = null;
      if (state.detection.ws) state.detection.ws.close();
      state.detection.ws = null;
      el.btnCam.textContent = "Start Webcam";
      el.statusText.textContent = "Webcam off";
      state.ui.frameCaptured = false;
      UI.updateCaptureButton();
      UI.updateGenerateButton();
    },

    toggle: function () {
      if (state.camera.stream) {
        Camera.stop();
      } else {
        Camera.start();
      }
    }
  };

  // =========================================================================
  // Detection — WebSocket connection, frame send/receive loop
  // =========================================================================
  var Detection = {
    connect: function () {
      var mode = UI.currentMode();
      var url = "/ws/detect?mode=" + encodeURIComponent(mode);
      if (mode === "object") {
        url += "&classes=" + encodeURIComponent(UI.currentClasses());
      }

      state.detection.ws = new WebSocket(Util.wsUrl(url));
      state.detection.ws.binaryType = "arraybuffer";

      state.detection.ws.onopen = function () {
        el.statusText.textContent = "Connected \u2014 streaming";
        state.detection.active = true;
        UI.updateCaptureButton();
        Detection.sendFrame();
      };

      state.detection.ws.onmessage = function (ev) {
        var resp = JSON.parse(ev.data);

        if (resp.error) {
          el.statusText.textContent = resp.error;
          state.detection.active = false;
          UI.updateCaptureButton();
          return;
        }

        Util.drawBase64(ctx.detect, el.canvasDetect, resp.detect);
        Util.drawBase64(ctx.mask, el.canvasMask, resp.mask);
        el.infoDetect.textContent = resp.label || (resp.count + " detected");

        state.detection.waitingForResponse = false;
        if (state.detection.active) {
          Detection.sendFrame();
        }
      };

      state.detection.ws.onclose = function () {
        state.detection.active = false;
        UI.updateCaptureButton();
        el.statusText.textContent = "Detect WS closed";
      };

      state.detection.ws.onerror = function () {
        state.detection.active = false;
        UI.updateCaptureButton();
        el.statusText.textContent = "Detect WS error";
      };
    },

    reconnect: function () {
      if (state.detection.ws) {
        state.detection.ws.onclose = null;
        state.detection.ws.onerror = null;
        state.detection.ws.close();
        state.detection.ws = null;
      }
      state.detection.waitingForResponse = false;
      state.detection.active = false;
      if (state.camera.stream) {
        el.statusText.textContent = "Loading model\u2026";
        el.infoDetect.textContent = "Switching\u2026";
        Detection.connect();
      }
    },

    sendFrame: function () {
      if (!state.detection.active || !state.detection.ws ||
          state.detection.ws.readyState !== WebSocket.OPEN) return;
      if (state.detection.waitingForResponse) return;

      el.captureCanvas.width = el.video.videoWidth || 640;
      el.captureCanvas.height = el.video.videoHeight || 480;
      ctx.capture.drawImage(el.video, 0, 0);

      el.captureCanvas.toBlob(
        function (blob) {
          if (blob && state.detection.ws &&
              state.detection.ws.readyState === WebSocket.OPEN) {
            state.detection.waitingForResponse = true;
            blob.arrayBuffer().then(function (buf) {
              state.detection.ws.send(buf);
            });
          }
        },
        "image/jpeg",
        0.75
      );
    }
  };

  // =========================================================================
  // Inpainting — WebSocket connection, generate/cancel, progress handling
  // =========================================================================
  var Inpainting = {
    connect: function () {
      state.inpainting.ws = new WebSocket(Util.wsUrl("/ws/inpaint"));

      state.inpainting.ws.onmessage = function (ev) {
        var msg = JSON.parse(ev.data);

        if (msg.error) {
          alert("Error: " + msg.error);
          state.inpainting.generating = false;
          UI.updateGenerateButton();
          el.progressContainer.hidden = true;
          return;
        }

        if (msg.status === "started") {
          el.resultLabel.textContent = "GENERATING\u2026";
          UI.showResultSection();
          el.progressContainer.hidden = false;
          el.progressBar.style.width = "0%";
          el.progressText.textContent = "Starting\u2026";
        } else if (msg.status === "progress") {
          var pct = Math.round((msg.step / msg.total_steps) * 100);
          el.progressBar.style.width = pct + "%";
          el.progressText.textContent = "Step " + msg.step + "/" + msg.total_steps + " (" + msg.elapsed + "s)";
        } else if (msg.status === "cancelled") {
          state.inpainting.generating = false;
          el.resultLabel.textContent = "CANCELLED";
          el.progressText.textContent = "Cancelled";
          UI.updateGenerateButton();
          setTimeout(function () { el.progressContainer.hidden = true; }, 2000);
        } else if (msg.status === "done") {
          el.progressBar.style.width = "100%";
          el.progressText.textContent = "Done in " + msg.elapsed + "s";
          Util.drawBase64(ctx.result, el.canvasResult, msg.image);
          el.resultLabel.textContent = "GENERATED";
          state.inpainting.generating = false;
          state.ui.hasGeneratedImage = true;
          el.btnDownload.disabled = false;
          UI.updateGenerateButton();
          setTimeout(function () { el.progressContainer.hidden = true; }, 3000);
        }
      };

      state.inpainting.ws.onclose = function () {
        state.inpainting.generating = false;
        UI.updateGenerateButton();
      };
    },

    generate: function () {
      if (state.inpainting.generating) {
        Inpainting.cancel();
        return;
      }

      var prompt = el.inputPrompt.value.trim();
      if (!prompt) {
        alert("Enter a prompt first.");
        return;
      }

      state.inpainting.generating = true;
      el.btnGen.disabled = false;
      el.btnGen.textContent = "Cancel";
      el.btnGen.classList.add("btn-cancel");

      if (!state.inpainting.ws || state.inpainting.ws.readyState !== WebSocket.OPEN) {
        Inpainting.connect();
        state.inpainting.ws.onopen = function () {
          state.inpainting.ws.send(JSON.stringify({ prompt: prompt }));
        };
      } else {
        state.inpainting.ws.send(JSON.stringify({ prompt: prompt }));
      }
    },

    cancel: function () {
      if (state.inpainting.ws && state.inpainting.ws.readyState === WebSocket.OPEN) {
        state.inpainting.ws.send(JSON.stringify({ action: "cancel" }));
      }
      el.btnGen.disabled = true;
      el.btnGen.textContent = "Cancelling\u2026";
    }
  };

  // =========================================================================
  // Model Status — poll /api/status for detector and inpaint readiness
  // =========================================================================
  var ModelStatus = {
    poll: function () {
      fetch("/api/status")
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var detectReady = data.detection === "ready";
          el.chipDetect.className = "status-chip " + (detectReady ? "ready" : "loading");

          if (data.inpaint === "ready") {
            el.chipInpaint.className = "status-chip ready";
            el.modelBanner.classList.add("hidden");
            state.inpainting.ready = true;
          } else if (data.inpaint === "error") {
            el.chipInpaint.className = "status-chip error";
            el.modelBanner.classList.add("hidden");
            state.inpainting.ready = false;
          } else {
            el.chipInpaint.className = "status-chip loading";
            el.bannerDetail.textContent = data.inpaint_detail || "Loading...";
            state.inpainting.ready = false;
          }
          UI.updateGenerateButton();

          if (data.inpaint !== "ready" && data.inpaint !== "error") {
            setTimeout(ModelStatus.poll, 3000);
          }
        })
        .catch(function () {
          setTimeout(ModelStatus.poll, 5000);
        });
    }
  };

  // =========================================================================
  // Capture — freeze current detection frame for inpainting
  // =========================================================================
  var Capture = {
    take: function () {
      el.btnCapture.disabled = true;
      el.btnCapture.textContent = "Capturing\u2026";

      fetch("/api/capture", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.error) {
            alert("Capture failed: " + data.error);
            el.btnCapture.disabled = false;
            el.btnCapture.textContent = "Capture Frame";
            return;
          }

          UI.showResultSection();
          Util.drawBase64(ctx.result, el.canvasResult, data.mask);
          el.resultLabel.textContent = "CAPTURED PREVIEW";
          state.ui.hasGeneratedImage = false;
          el.btnDownload.disabled = true;

          state.ui.frameCaptured = true;
          el.btnCapture.textContent = "Recapture";
          el.btnCapture.disabled = false;
          UI.updateGenerateButton();
        })
        .catch(function () {
          alert("Capture request failed.");
          el.btnCapture.disabled = false;
          el.btnCapture.textContent = "Capture Frame";
        });
    }
  };

  // =========================================================================
  // Event Bindings
  // =========================================================================
  el.modeSelect.addEventListener("change", function () {
    UI.updateModeUI();
    Detection.reconnect();
  });

  el.classSelect.addEventListener("change", function () {
    Detection.reconnect();
  });

  el.btnCam.addEventListener("click", Camera.toggle);
  el.btnCapture.addEventListener("click", Capture.take);
  el.btnGen.addEventListener("click", Inpainting.generate);

  el.btnDownload.addEventListener("click", function () {
    if (!state.ui.hasGeneratedImage) return;
    var link = document.createElement("a");
    link.download = "inpainted-" + Date.now() + ".png";
    link.href = el.canvasResult.toDataURL("image/png");
    link.click();
  });

  // =========================================================================
  // Initialization
  // =========================================================================
  UI.updateModeUI();
  ModelStatus.poll();

  navigator.mediaDevices.getUserMedia({ video: true }).then(function (tmp) {
    tmp.getTracks().forEach(function (t) { t.stop(); });
    Camera.populateDevices();
  }).catch(function () {
    Camera.populateDevices();
  });
})();
