(function () {
  const storageKey = "dashboardAudioEnabled";
  const alertThrottleMs = 3000;
  let audioContext = null;
  let status = "disabled";
  let muted = false;
  let unlocked = false;
  let lastAlertSoundAt = 0;
  let autoUnlockBound = false;

  function audioCtor() {
    return window.AudioContext || window.webkitAudioContext || null;
  }

  function statusText() {
    return `Sound: ${status}`;
  }

  function updateStatusUi() {
    const statusNode = document.getElementById("demo-sound-status");
    const enableButton = document.getElementById("demo-enable-sound");
    const muteButton = document.getElementById("demo-mute-sound");
    if (statusNode) {
      statusNode.textContent = statusText();
      statusNode.className = `sound-status sound-${status}`;
    }
    if (enableButton) {
      enableButton.textContent = status === "enabled" ? "Sound enabled" : "Enable sound";
    }
    if (muteButton) {
      muteButton.textContent = muted ? "Unmute sound" : "Mute sound";
    }
  }

  function setStatus(value) {
    status = value;
    updateStatusUi();
    return status;
  }

  function removeAutoUnlockHandlers() {
    document.removeEventListener("pointerdown", handleFirstUserGesture, true);
    document.removeEventListener("keydown", handleFirstUserGesture, true);
    document.removeEventListener("touchstart", handleFirstUserGesture, true);
    autoUnlockBound = false;
  }

  function bindAutoUnlockHandlers() {
    if (autoUnlockBound) {
      return;
    }
    document.addEventListener("pointerdown", handleFirstUserGesture, true);
    document.addEventListener("keydown", handleFirstUserGesture, true);
    document.addEventListener("touchstart", handleFirstUserGesture, true);
    autoUnlockBound = true;
  }

  function handleFirstUserGesture() {
    unlock().then((unlockStatus) => {
      console.info(`LiveLabAudio first interaction unlock status: ${unlockStatus}`);
    });
  }

  function init() {
    if (!audioCtor()) {
      console.info("LiveLabAudio: Web Audio API unsupported");
      return setStatus("unsupported");
    }
    window.localStorage.setItem(storageKey, "true");
    bindAutoUnlockHandlers();
    console.debug("LiveLabAudio init: always armed; waiting for browser unlock if needed");
    setStatus("blocked");
    unlock().then((unlockStatus) => {
      if (unlockStatus === "blocked") {
        console.info("LiveLabAudio auto unlock blocked; click anywhere on the dashboard to enable sound");
      }
    });
    return status;
  }

  async function ensureContext() {
    const AudioCtor = audioCtor();
    if (!AudioCtor) {
      return null;
    }
    if (!audioContext) {
      audioContext = new AudioCtor();
    }
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    return audioContext;
  }

  async function unlock() {
    if (!audioCtor()) {
      console.info("LiveLabAudio unlock: unsupported");
      return setStatus("unsupported");
    }
    try {
      const ctx = await ensureContext();
      if (!ctx || ctx.state !== "running") {
        console.info(`LiveLabAudio unlock blocked: state=${ctx ? ctx.state : "none"}`);
        return setStatus("blocked");
      }
      unlocked = true;
      muted = false;
      window.localStorage.setItem(storageKey, "true");
      removeAutoUnlockHandlers();
      console.info("LiveLabAudio unlock status: enabled");
      return setStatus("enabled");
    } catch (error) {
      console.info(`LiveLabAudio unlock blocked: ${error}`);
      return setStatus("blocked");
    }
  }

  function isEnabled() {
    return status === "enabled" && unlocked && !muted;
  }

  function getStatus() {
    return status;
  }

  function canPlay() {
    if (!unlocked || status !== "enabled") {
      console.debug(`LiveLabAudio blocked/disabled: status=${status}, unlocked=${unlocked}`);
      return false;
    }
    if (muted) {
      console.debug("LiveLabAudio muted");
      return false;
    }
    return true;
  }

  function scheduleTone(ctx, frequency, start, duration, volume) {
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(frequency, ctx.currentTime + start);
    gain.gain.setValueAtTime(0.0001, ctx.currentTime + start);
    gain.gain.exponentialRampToValueAtTime(volume, ctx.currentTime + start + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + start + duration);
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.start(ctx.currentTime + start);
    oscillator.stop(ctx.currentTime + start + duration + 0.03);
  }

  function playSequence(notes, volume) {
    if (!canPlay()) {
      return false;
    }
    const ctx = audioContext;
    if (!ctx || ctx.state !== "running") {
      console.info("LiveLabAudio blocked/unsupported while playing");
      setStatus(ctx ? "blocked" : "unsupported");
      return false;
    }
    notes.forEach((note) => scheduleTone(ctx, note.frequency, note.start, note.duration, volume));
    return true;
  }

  function playNodeConnected() {
    const played = playSequence([{ frequency: 520, start: 0.0, duration: 0.15 }], 0.035);
    if (played) {
      console.debug("Node sound triggered");
    }
    return played;
  }

  function canPlayAlertSound() {
    const now = Date.now();
    if (now - lastAlertSoundAt < alertThrottleMs) {
      return false;
    }
    lastAlertSoundAt = now;
    return true;
  }

  function playAttackAlert() {
    if (!canPlayAlertSound()) {
      return false;
    }
    const played = playSequence(
      [
        { frequency: 880, start: 0.0, duration: 0.12 },
        { frequency: 1040, start: 0.18, duration: 0.16 },
      ],
      0.06
    );
    if (played) {
      console.debug("Alert sound triggered");
    }
    return played;
  }

  function toggleMute() {
    muted = !muted;
    if (muted && status === "enabled") {
      console.debug("LiveLabAudio muted by user");
    }
    updateStatusUi();
    return muted;
  }

  window.LiveLabAudio = {
    init,
    unlock,
    playNodeConnected,
    playAttackAlert,
    isEnabled,
    getStatus,
    toggleMute,
  };
})();
