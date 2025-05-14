$(document).ready(function () {
  console.log('✅ DOM pronto e script caricato');

  let selectedIt = null;
  let selectedEn = null;

  // Ripristina l’input principale
  const saved = localStorage.getItem('savedUserInput');
  if (saved) {
    console.log('📥 Ripristinato da localStorage:', saved);
    $('#userInput').val(saved);
  }

  // Funzione di setup per IT/EN
  function setupSelection(selector, setter) {
    $(selector).on('click', 'input', function () {
      const $this = $(this);
      const was = $this.hasClass('selected');
      console.log(`🖱️ Click su ${selector}:`, $this.val());

      // Deseleziona tutto
      $(selector + ' input')
        .removeClass('selected')
        .prop('readonly', true);

      if (!was) {
        $this
          .addClass('selected')
          .prop('readonly', false)
          .focus();
        setter($this);
      } else {
        setter(null);
      }
      checkReady();
    });
  }

  // Abilita/disabilita submit
  function checkReady() {
    const ready = Boolean(selectedIt && selectedEn);
    console.log('🧪 checkReady → IT:', selectedIt?.val(), 'EN:', selectedEn?.val(), '→ ready=', ready);
    $('#createBtn').prop('disabled', !ready);
  }

  // Inizializza
  setupSelection('#descIt', it => selectedIt = it);
  setupSelection('#descEn', en => selectedEn = en);

  // Submit unico
  $('#createBtn').on('click', function () {
    console.log('🖱️ submitBtn cliccato');
    if (!(selectedIt && selectedEn)) {
      console.warn('❌ Mancano selezioni!');
      return;
    }
    // Salva input principale
    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);
    console.log('📥 Salvato main input:', main);

    // Prepara payload
    const payload = {
      code: $('#materialCode').val().trim(),
      desc_it: selectedIt.val().trim(),
      desc_en: selectedEn.val().trim()
    };
    console.log('📤 Payload:', payload);

    // AJAX
    $.ajax({
      method: 'POST',
      url: '/submit_insertion',
      contentType: 'application/json',
      data: JSON.stringify(payload),
    })
    .done(resp => {
      console.log('✅ Response:', resp);
      alert('Inserimento OK');
      // window.location.href = '/';
    })
    .fail((xhr, status, err) => {
      console.error('❌ Errore AJAX:', status, err);
      alert('Errore invio');
    });
  });

  // Cancel
  $('#cancelBtn').on('click', function () {
    console.log('🖱️ cancelBtn cliccato');
    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);
    console.log('📥 Salvato main input:', main);
    window.location.href = '/';
  });
});
