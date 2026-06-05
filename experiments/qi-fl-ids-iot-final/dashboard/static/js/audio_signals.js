(function () {
  const storageKey = "dashboardAudioEnabled";
  const alertThrottleMs = 3000;
  let audioContext = null;
  let audioEnabled = window.localStorage.getItem(storageKey) === "true";
  let muted = false;
  let lastAlertSoundAt = 0;

  function context() {
    if (!audioContext) {
      const AudioCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtor) {
        return null;
      }
      audioContext = new AudioCtor();
    }
    if (audioContext.state === "suspended") {
      audioContext.resume().catch(() => {});
    }
    return audioContext;
  }

  function setButtonState() {
    const enable = document.getElementById("demo-enable-sound");
    const mute = document.getElementById("demo-mute-sound");
    if (enable) {
      enable.textContent = audioEnabled ? "Sound enabled" : "Enable sound";
    }
    if (mute) {
      mute.textContent = muted ? "Unmute sound" : "Mute sound";
    }
  }

  function setAudioEnabled(value) {
    audioEnabled = Boolean(value);
    window.localStorage.setItem(storageKey, audioEnabled ? "true" : "false");
    if (audioEnabled) {
      context();
    }
    setButtonState();
  }

  function toggleMute() {
    muted = !muted;
    setButtonState();
  }

  function canPlay() {
    return audioEnabled && !muted;
  }

  function beepSequence(notes, baseVolume) {
    if (!canPlay()) {
      return;
    }
    const ctx = context();
    if (!ctx) {
      return;
    }
    const now = ctx.currentTime;
    notes.forEach((note) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(note.frequency, now + note.start);
      gain.gain.setValueAtTime(0.0001, now + note.start);
      gain.gain.exponentialRampToValueAtTime(baseVolume, now + note.start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + note.start + note.duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + note.start);
      osc.stop(now + note.start + note.duration + 0.03);
    });
  }

  function playDeviceConnectedSound() {
    beepSequence(
      [
        { frequency: 660, start: 0.0, duration: 0.12 },
        { frequency: 880, start: 0.15, duration: 0.14 },
      ],
      0.035
    );
  }

  function canPlayAlertSound() {
    const now = Date.now();
    if (now - lastAlertSoundAt < alertThrottleMs) {
      return false;
    }
    lastAlertSoundAt = now;
    return true;
  }

  function playAttackDetectedSound() {
    if (!canPlay()) {
      return;
    }
    if (!canPlayAlertSound()) {
      return;
    }
    beepSequence(
      [
        { frequency: 880, start: 0.0, duration: 0.11 },
        { frequency: 440, start: 0.18, duration: 0.11 },
        { frequency: 880, start: 0.36, duration: 0.13 },
      ],
      0.06
    );
  }

  function bindControls() {
    const enable = document.getElementById("demo-enable-sound");
    const mute = document.getElementById("demo-mute-sound");
    if (enable) {
      enable.addEventListener("click", () => setAudioEnabled(true));
    }
    if (mute) {
      mute.addEventListener("click", toggleMute);
    }
    setButtonState();
  }

  window.P1618Audio = {
    bindControls,
    canPlayAlertSound,
    playAttackDetectedSound,
    playDeviceConnectedSound,
    setAudioEnabled,
    toggleMute,
  };
})();
