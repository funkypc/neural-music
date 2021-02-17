// Change the button text on form submit
function changeButton() {
// disable button
$("#submit").prop("disabled", true);
// add spinner to button
$("#submit").html(
'<i class="fa fa-circle-o-notch fa-spin"></i> Generating...'
);
};