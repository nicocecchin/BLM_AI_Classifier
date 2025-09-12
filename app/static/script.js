$(document).ready(function () {
  // --- Translations: add keys as needed ---
  const translations = {
    it: {
      placeholder: "Inserisci testo...",
      submit: "Invia",
      createNew: "Crea nuovo",
      ok: "OK",
      loading: "Caricamento...",
      errorRequest: "Errore nella richiesta al server.",
      selectedAlert: "Hai selezionato:"
    },
    en: {
      placeholder: "Insert text...",
      submit: "Submit",
      createNew: "Create new",
      ok: "OK",
      loading: "Loading...",
      errorRequest: "Error on server request.",
      selectedAlert: "You selected:"
    }
  };

  // --- Language initialization ---
  let selectedLang = localStorage.getItem('lang') || 'it';
  const $langToggle = $('#langToggle');

  function applyLang(lang) {
    selectedLang = lang;
    localStorage.setItem('lang', lang);
    // update flag icon (emoji)
    $langToggle.text(lang === 'it' ? '🇮🇹' : '🇬🇧');
    // update HTML lang attribute for accessibility / screen readers
    document.documentElement.lang = lang === 'it' ? 'it' : 'en';
    // update placeholder/button texts
    $('#userInput').attr('placeholder', translations[lang].placeholder);
    $('#submitBtn').text(translations[lang].submit);
    $('#createNewBtn').text(translations[lang].createNew); // if you have this button in content
    $('#okBtn').text(translations[lang].ok); // if you have this button in content
    $('#loadingSpannable').text(translations[lang].loading);
  }

  // initial apply
  applyLang(selectedLang);

  // toggle on click
  $langToggle.on('click', function () {
    applyLang(selectedLang === 'it' ? 'en' : 'it');
  });


  // --- saved input ---
  const saved = localStorage.getItem('savedUserInput');
  if (saved) {
    $('#userInput').val(saved);
  }

  // --- helper to pick description by language ---
  function formattedDescription(item, lang) {
    // show main in chosen lang, secondary in the other
    if (lang === 'it') {
      return `<strong>${item.description_ita}</strong> — <em class="text-muted">${item.description_eng}</em>`;
    } else {
      return `<strong>${item.description_eng}</strong> — <em class="text-muted">${item.description_ita}</em>`;
    }
  }

  // --- render results ---
  let selectedItem = null;
  function renderResults(items) {
    $('#results').empty();
    selectedItem = null;
    $('#okBtn').prop('disabled', true);

    items.forEach(item => {
      const $item = $(`
        <div class="border rounded p-2 mb-2 result-item" style="cursor: pointer;">
          <div class="d-flex justify-content-between align-items-start">
            <small class="text-muted">ID: ${item.id}</small>
            <span class="badge bg-info text-dark">Score: ${item.score}</span>
          </div>
          <div class="text-truncate small mt-1">
            ${formattedDescription(item, selectedLang)}
          </div>
        </div>
      `);

      $item.on('click', function (e) {
        e.stopPropagation();
        const alreadySelected = $(this).hasClass('border-primary');

        $('.result-item').removeClass('border-primary');
        selectedItem = null;
        $('#okBtn').prop('disabled', true);

        if (!alreadySelected) {
          $(this).addClass('border-primary');
          selectedItem = item;
          $('#okBtn').prop('disabled', false);
        }
      });

      $('#results').append($item);
    });
  }

  // --- send AJAX with lang included ---
  function sendRequest(input) {
    $('#results').empty();
    $('#loadingSpinner').css('display', 'block');

    $.ajax({
      url: '/get_results',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ input: input, lang: selectedLang }),
      success: function (data) {
        // data assumed to be an array of items with description_ita and description_eng
        renderResults(data);
        console.log(data);
      },
      error: function () {
        alert(translations[selectedLang].errorRequest);
      },
      complete: function() {
        $('#loadingSpinner').css('display', 'none');
      }
    });
  }

  // --- events for sending input ---
  $('#submitBtn').on('click', function () {
    const input = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', input);
    if (input) sendRequest(input);
  });

  $(document).on('keydown', function(event) {
    if (event.key === 'Enter') {
      const input = $('#userInput').val().trim();
      localStorage.setItem('savedUserInput', input);
      if (input) sendRequest(input);
    }
  });

  // Deseleziona tutto cliccando fuori
  $(document).on('click', function () {
    $('.result-item').removeClass('active border-primary');
    selectedItem = null;
    $('#okBtn').prop('disabled', true);
  });

  // Click su OK
  $('#okBtn').on('click', function () {
    if (selectedItem) {
      // show a friendly message in the chosen language
      const mainDesc = selectedLang === 'it' ? selectedItem.description_ita : selectedItem.description_eng;
      alert(`${translations[selectedLang].selectedAlert} ${mainDesc}`);
    }
  });

  // Click su Create New (redirect)
  $('#createNewBtn').on('click', function () {
    const input = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', input);
    // preserve language in the query string if your insertion page wants it
    window.location.href = '/insertion' + '?lang=' + selectedLang;
  });

  // Abilita/disabilita "Create New" in base all'input utente
  $('#userInput').on('input', function () {
    const isNotEmpty = $(this).val().trim().length > 0;
    $('#createNewBtn').prop('disabled', !isNotEmpty);
  });

  // Trigger iniziale per stato corretto su load
  $('#userInput').trigger('input');

});
