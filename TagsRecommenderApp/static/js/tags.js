$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				title: $('#title').val(),
				pagedown_text : $('#box').val()
			},
			type : 'POST',
			url : '/result'
		})
		.done(function(data) {
            $('#successAlert').text(data.rec_tags).show();

		});

		event.preventDefault();

	});

});
