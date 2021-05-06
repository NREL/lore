// Specify which menu button gets highlighted based on path
$(".sidebar-navigation li > a").each(function() {
    if ((window.location.pathname == $(this).attr('href'))) {
        $(this).parent().addClass('active');
    }
});