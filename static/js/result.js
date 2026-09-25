document.querySelectorAll('[data-action="print"]').forEach(function (button) {
  button.addEventListener('click', function () { window.print(); });
});
