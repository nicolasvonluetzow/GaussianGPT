window.HELP_IMPROVE_VIDEOJS = false;

function initBeforeAfterSlider(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return;

  var before = container.querySelector('.bal-before');
  var beforeInset = container.querySelector('.bal-before-inset');
  var handle = container.querySelector('.bal-handle');
  var afterImg = container.querySelector('.bal-after img');

  var isDragging = false;

  function setPosition(pct) {
    pct = Math.max(0, Math.min(100, pct));
    before.style.width = pct + '%';
    beforeInset.style.width = container.offsetWidth + 'px';
    handle.style.left = pct + '%';
  }

  function getClientX(e) {
    return e.touches ? e.touches[0].clientX : e.clientX;
  }

  function onMove(e) {
    if (!isDragging) return;
    var rect = container.getBoundingClientRect();
    setPosition(((getClientX(e) - rect.left) / rect.width) * 100);
  }

  function onUp() { isDragging = false; }
  function onDown(e) { isDragging = true; e.preventDefault(); }

  container.addEventListener('mousedown', onDown);
  container.addEventListener('touchstart', onDown, { passive: false });
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove, { passive: false });
  document.addEventListener('mouseup', onUp);
  document.addEventListener('touchend', onUp);

  window.addEventListener('resize', function() { setPosition(50); });

  function init() {
    beforeInset.style.width = container.offsetWidth + 'px';
    setPosition(50);
  }

  if (afterImg.complete) { init(); } else { afterImg.addEventListener('load', init); }
}

$(document).ready(function() {
    $(".navbar-burger").click(function() {
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    initBeforeAfterSlider('example4');
    initBeforeAfterSlider('example5');
    initBeforeAfterSlider('example6');
    initBeforeAfterSlider('example7');
})
