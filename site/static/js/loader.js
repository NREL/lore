// Get div.data element
const targetNode = $('.data* .data')[0];

// Get intended Bokeh chart count
const totalChartCount = $(targetNode).find('.card-body').length;

// Create variable to track count for loaded charts
var chartsLoaded = 0;

// Set config
const config = {childList:true, subtree: true};

// Remove Loader when all charts are loaded
const removeLoader = function(mutationList, observer){
    for (var mutation of mutationList){
        var target = $(mutation.target);
        if(target.hasClass('bk-root')){
            if(target.find('.bk').length > 0){
               chartsLoaded += 1; 
            }
            if(chartsLoaded == totalChartCount){
                // Hide modal once all charts are loaded
                $('.loader-modal').hide();
                $('svg#expand').show();
            }
            
        }
    }

}

if (totalChartCount == 0){ // If there are no Bokeh charts, remove the modal
                           // NOTE: This will need to change once maintenance and settings pages are implemented
    $('.loader-modal').hide()
}
else{
    // Create the observer
    const observer = new MutationObserver(removeLoader);

    // Start observing the target node
    observer.observe(targetNode, config);
}