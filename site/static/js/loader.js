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
                $('.fa-expand').show();
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

// MODAL CODE
// Get the image and insert it inside the modal - use its "alt" text as a caption
var modal = document.getElementById('chart-modal');
var modalContent = document.getElementById("bokeh-chart");
var captionText = document.getElementById("caption");

// Get the <span> element that closes the modal
var closeX = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x) or hits ESC, close the modal
closeX.onclick = close;
$(document).on('keydown', function(e){
    if(e.key == 'Escape'){
        close();
    }
})
function close() { 
  modal.style.display = "none";

  // Remove modal content
  var modalContent = $('.modal-content');
  modalContent.remove();
}

function openModal(el){

  // Add Bokeh Script to modal content
  var scriptElement = $(el.parentElement).find('#plot_script')[0].value;
  modalContent = document.createElement('div');
  modalContent.classList.add('modal-content');
  modalContent.id = 'bokeh-chart';
  modalContent.innerHTML = scriptElement;
  $('#chart-modal')[0].appendChild(modalContent);

  // Execute script in .modal-content
  var script = $('#chart-modal').find('script')[0];

  // Show modal
  modal.style.display = "block";

  // Timeout to let modal get to full size
  setTimeout(eval(script.text), 500);
  
//   captionText.innerHTML = this.alt;
}