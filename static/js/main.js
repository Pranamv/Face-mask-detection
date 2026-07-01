document.addEventListener('DOMContentLoaded', () => {
  const imgForm = document.getElementById('image-form');
  const imgResult = document.getElementById('image-result');

  if (imgForm && imgResult) {
    imgForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(imgForm);
      imgResult.classList.remove('hidden');
      imgResult.innerHTML = '<div class="spinner">Analyzing image...</div>';
      try {
        const resp = await fetch(imgForm.action, { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Request failed');
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        imgResult.innerHTML = `<img src="${url}" alt="Result" style="max-width:100%;border-radius:12px;"/>`;
      } catch (err) {
        imgResult.innerHTML = '<div class="error">Failed to analyze image. Please try again.</div>';
      }
    });
  }

  // Live page: save faces without mask
  const captureBtn = document.getElementById('capture-toggle-btn');
  const captureStatus = document.getElementById('capture-status');
  if (captureBtn && captureStatus) {
    const refreshStatus = async () => {
      try {
        const r = await fetch('/capture_status');
        const d = await r.json();
        if (d.enabled) {
          captureBtn.textContent = 'Stop Capturing (No-Mask)';
          captureStatus.textContent = 'Capturing no-mask faces...';
        } else {
          captureBtn.textContent = 'Start Capturing (No-Mask)';
          captureStatus.textContent = 'Not capturing';
        }
      } catch {}
    };

    captureBtn.addEventListener('click', async () => {
      captureBtn.disabled = true;
      try {
        const r = await fetch('/toggle_capture', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
        await r.json();
        await refreshStatus();
      } catch {}
      captureBtn.disabled = false;
    });

    // Poll status every 2 seconds to keep UI in sync
    setInterval(refreshStatus, 2000);
    // Initial
    refreshStatus();
  }

  // Show selected file name for image/video inputs
  const fileInputs = document.querySelectorAll('.file-input input[type="file"]');
  fileInputs.forEach(inp => {
    inp.addEventListener('change', () => {
      const span = inp.parentElement?.querySelector('span');
      const form = inp.closest('form');
      const clearBtn = form ? form.querySelector('.clear-file') : null;
      if (!span) return;
      const name = inp.files && inp.files.length > 0 ? inp.files[0].name : 'Select a file...';
      span.textContent = name;
      // Toggle clear button visibility
      if (clearBtn) {
        if (inp.files && inp.files.length > 0) {
          clearBtn.classList.remove('hidden');
        } else {
          clearBtn.classList.add('hidden');
        }
      }
    });
  });

  // Clear selected file when X button is clicked
  document.querySelectorAll('.clear-file').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const form = btn.closest('form');
      if (!form) return;
      const fileInput = form.querySelector('.file-input input[type="file"]');
      const span = form.querySelector('.file-input span');
      if (fileInput) {
        fileInput.value = '';
      }
      if (span) {
        // Pick placeholder by input name
        const isVideo = (fileInput && fileInput.getAttribute('name') === 'video');
        span.textContent = isVideo ? 'Select a video...' : 'Select an image...';
      }
      // Hide the clear button again
      btn.classList.add('hidden');
    });
  });
});
