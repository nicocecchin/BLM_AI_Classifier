$(document).ready(function () {
  console.log('✅ DOM pronto e script caricato');

  const translations = {
    it: {
      title: "Inserimento nuovo materiale",
      labelDescIt: "Descrizione breve (Italiano)",
      labelDescEn: "Descrizione breve (Inglese)",
      createBtn: "Inserisci",
      cancelBtn: "Annulla",
      loading: "Caricamento...",
      suggestionError: "Errore nella richiesta di suggerimenti",
      submitOk: "Inserimento OK",
      submitError: "Errore invio"
    },
    en: {
      title: "Insert new material",
      labelDescIt: "Short description (Italian)",
      labelDescEn: "Short description (English)",
      createBtn: "Submit",
      cancelBtn: "Cancel",
      loading: "Loading...",
      suggestionError: "Error with suggestion request",
      submitOk: "Insertion OK",
      submitError: "Error on submission"
    }
  };

  // language
  let selectedLang = localStorage.getItem('lang') || 'it';
  function applyLang(lang) {
    if (!lang) lang = 'it';
    selectedLang = lang;
    $('#insertionTitle').text(translations[lang].title);
    $('#labelDescIt').text(translations[lang].labelDescIt);
    $('#labelDescEn').text(translations[lang].labelDescEn);
    $('#createBtn').text(translations[lang].createBtn);
    $('#cancelBtn').text(translations[lang].cancelBtn);
    document.documentElement.lang = (lang === 'it') ? 'it' : 'en';
  }
  applyLang(selectedLang);
  $(document).on('click', '#langToggle', function () {
    setTimeout(() => applyLang(localStorage.getItem('lang') || 'it'), 25);
  });
  window.addEventListener('storage', function (e) {
    if (e.key === 'lang') applyLang(e.newValue || 'it');
  });

  // state
  let selectedIt = null;
  let selectedEn = null;

  // restore main input
  const saved = localStorage.getItem('savedUserInput');
  if (saved) $('#userInput').val(saved);

  // utils
  function escapeHtml(str) {
    return String(str === undefined || str === null ? '' : str)
      .replaceAll('&','&amp;')
      .replaceAll('<','&lt;')
      .replaceAll('>','&gt;')
      .replaceAll('"','&quot;')
      .replaceAll("'",'&#39;');
  }

  // Create ONE single box per suggestion: a card with border, input (borderless) + counter
  function createSuggestionItem(text) {
    const safeText = escapeHtml(text);
    const $card = $(`
      <div class="suggestion-card d-flex align-items-center mb-2 p-2 border rounded">
        <input type="text" class="suggestion-input flex-grow-1 me-2" value="${safeText}" readonly aria-label="suggestion">
        <small class="char-count text-muted ms-1" aria-hidden="true">0</small>
      </div>
    `);
    const $input = $card.find('.suggestion-input');
    const $counter = $card.find('.char-count');
    $counter.text($input.val().length);
    return $card;
  }

  // selection handling (delegated)
  function setupSelection(selector, setter) {
    $(selector).on('click', '.suggestion-input', function (e) {
      e.stopPropagation();
      const $input = $(this);
      const $card = $input.closest('.suggestion-card');
      const was = $input.hasClass('selected');

      // deselect all in group
      $(selector + ' .suggestion-input').removeClass('selected').prop('readonly', true);
      $(selector + ' .suggestion-card').removeClass('selected-card');

      if (!was) {
        $input.addClass('selected').prop('readonly', false).focus().select();
        $card.addClass('selected-card');
        setter($input);
      } else {
        $input.removeClass('selected').prop('readonly', true);
        $card.removeClass('selected-card');
        setter(null);
      }
      checkReady();
    });
  }

  function checkReady() {
    const ready = Boolean(selectedIt && selectedEn);
    $('#createBtn').prop('disabled', !ready);
  }

  // show suggestions
  function showSuggestions(it_suggestions, en_suggestions) {
    $('#descIt').empty();
    $('#descEn').empty();

    it_suggestions.forEach(el => $('#descIt').append(createSuggestionItem(el)));
    en_suggestions.forEach(el => $('#descEn').append(createSuggestionItem(el)));

    selectedIt = null;
    selectedEn = null;
    checkReady();
  }

  // update counters on input
  $('#descIt, #descEn').on('input', '.suggestion-input', function () {
    const $input = $(this);
    const len = ($input.val() || '').length;
    $input.closest('.suggestion-card').find('.char-count').text(len);
  });

  // loading
  function loading(){
    $('#descIt').empty();
    $('#descEn').empty();
    const tpl = `
      <div class="d-flex align-items-center py-2">
        <div class="spinner-border text-primary me-2" role="status" aria-hidden="true"></div>
        <small class="text-muted">${translations[selectedLang].loading}</small>
      </div>
    `;
    $('#descIt').append(tpl);
    $('#descEn').append(tpl);
  }

  setupSelection('#descIt', it => { selectedIt = it; });
  setupSelection('#descEn', en => { selectedEn = en; });

  // request suggestions
  function requestSuggestions() {
    const input = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', input);
    if (!input) return;

    loading();

    $.ajax({
      url: '/get_suggestions',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ input: input, lang: selectedLang }),
      success: function (data) {
        const { ita = [], eng = [] } = data || {};
        showSuggestions(ita, eng);
      },
      error: function () {
        alert(translations[selectedLang].suggestionError);
        $('#descIt').empty(); $('#descEn').empty();
      }
    });
  }

  $('#submitBtn').on('click', requestSuggestions);
  $(document).on('keydown', function(event) {
    if (event.key === 'Enter') requestSuggestions();
  });

  // final submit
  $('#createBtn').on('click', function () {
    if (!(selectedIt && selectedEn)) return;

    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);

    const payload = {
      code: ($('#materialCode').length ? $('#materialCode').val().trim() : null),
      desc_it: selectedIt.val().trim(),
      desc_en: selectedEn.val().trim(),
      lang: selectedLang,
      main_input: main
    };

    $.ajax({
      method: 'POST',
      url: '/submit_insertion',
      contentType: 'application/json',
      data: JSON.stringify(payload),
    })
    .done(() => {
      alert(translations[selectedLang].submitOk);
      window.location.href = '/';
    })
    .fail(() => {
      alert(translations[selectedLang].submitError);
    });
  });

  // cancel
  $('#cancelBtn').on('click', function () {
    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);
    window.location.href = '/';
  });

  // click outside -> deselect (remove selected-card border)
  $(document).on('click', function (e) {
    if (!$(e.target).closest('#descIt, #descEn').length) {
      $('#descIt .suggestion-input, #descEn .suggestion-input').removeClass('selected').prop('readonly', true);
      $('#descIt .suggestion-card, #descEn .suggestion-card').removeClass('selected-card');
      selectedIt = null; selectedEn = null; checkReady();
    }
  });

});
