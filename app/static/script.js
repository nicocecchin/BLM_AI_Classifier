$(document).ready(function () {
  let selectedItem = null;

  const saved = localStorage.getItem('savedUserInput');
    if (saved) {
        $('#userInput').val(saved);
        
    }


  // Funzione per mostrare i risultati
  function renderResults(items) {
    $('#results').empty();
    selectedItem = null;
    $('#okBtn').prop('disabled', true);

    items.forEach(itemText => {
      const $item = $('<button>')
        .addClass('list-group-item list-group-item-action result-item')
        .text(itemText);

      $item.on('click', function (e) {
        e.stopPropagation();
        const alreadySelected = $(this).hasClass('active');

        $('.result-item').removeClass('active');
        selectedItem = null;
        $('#okBtn').prop('disabled', true);

        if (!alreadySelected) {
          $(this).addClass('active');
          selectedItem = itemText;
          $('#okBtn').prop('disabled', false);
        }
      });

      $('#results').append($item);
    });
  }

  // Invio input con jQuery AJAX
  $('#submitBtn').on('click', function () {
    const input = $('#userInput').val().trim();
    if (input) {
      $.ajax({
        url: '/get_results',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ input: input }),
        success: function (data) {
          renderResults(data);
        },
        error: function () {
          alert('Errore nella richiesta al server.');
        }
      });
    }
  });

  // Deseleziona tutto cliccando fuori
  $(document).on('click', function () {
    $('.result-item').removeClass('active');
    selectedItem = null;
    $('#okBtn').prop('disabled', true);
  });

  // Click su OK
  $('#okBtn').on('click', function () {
    if (selectedItem) {
      alert(`Hai selezionato: ${selectedItem}`);
    }
  });

  // Click su Create New
  $('#createNewBtn').on('click', function () {
      const input = $('#userInput').val().trim();
      localStorage.setItem('savedUserInput', input);
      window.location.href = '/insertion';
  });

    // Abilita/disabilita "Create New" in base all'input utente
  $('#userInput').on('input', function () {
    const isNotEmpty = $(this).val().trim().length > 0;
    $('#createNewBtn').prop('disabled', !isNotEmpty);
  });

  // Trigger iniziale per stato corretto su load
  $('#userInput').trigger('input');

});
