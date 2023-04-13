// =================================================================================================
// =================================================================================================

// Declarations
var canvas = document.getElementById("paint");
var ctx = canvas.getContext("2d");
canvas.style.touchAction = "none";
var touchpad_option = true;
var width = canvas.width;
var height = canvas.height;
var curX, curY, prevX, prevY;
var hold = false;
ctx.lineWidth = 1;
document.getElementById("brushsize").innerHTML = ctx.lineWidth;
var fill_value = true;
var stroke_value = false;
var canvas_data = {"pencil": [], "eraser": []}
let isMale = false;

// =================================================================================================
// =================================================================================================

function color(color_value){
    ctx.strokeStyle = color_value;
    ctx.fillStyle = color_value;
} 

// =================================================================================================
// =================================================================================================
// canvas add brush pixel function
function add_brush(){
    if (ctx.lineWidth == 3){
        ctx.lineWidth = 3;
    }
    else{
        ctx.lineWidth += 1;
    }
    console.log("Added : ",ctx.lineWidth);
    document.getElementById("brushsize").innerHTML = ctx.lineWidth;
}

// =================================================================================================
// =================================================================================================

// canvas reduce brush pixel function
function reduce_brush(){
    if (ctx.lineWidth == 1){
        ctx.lineWidth = 1;
    }
    else{
        ctx.lineWidth -= 1;
    }
    console.log("Reduced : ",ctx.lineWidth);
    document.getElementById("brushsize").innerHTML = ctx.lineWidth;
}

// =================================================================================================
// =================================================================================================

// canvas reset function
function reset(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas_data = {"pencil": [], "eraser": []}
    ctx.globalAlpha = 0.1; // set opacity to 0.5

    const shadow = document.getElementById("shadow");
    // shadow.style.backgroundImage = 'none';
    if (isMale==true){
        shadow.style.backgroundImage = "url('/static/images/male_image.png')";
    }
    else{
        shadow.style.backgroundImage = "url('/static/images/female_image.png')";
    }
    // console.log("reset here");
}
   
// =================================================================================================
// =================================================================================================

canvas.addEventListener("touchstart", function(e){
    e.preventDefault();
    var touch = e.touches[0];
    curX = touch.clientX - canvas.offsetLeft;
    curY = touch.clientY - canvas.offsetTop;
    hold = true;
    prevX = curX;
    prevY = curY;
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
});

// =================================================================================================
// =================================================================================================

// Add event listener for touchmove
canvas.addEventListener("touchmove", function(e){
    e.preventDefault();
    if(hold){
        var touch = e.touches[0];
        curX = touch.clientX - canvas.offsetLeft;
        curY = touch.clientY - canvas.offsetTop;

        if ( touchpad_option == true ) {
            draw();

            function draw(){
                ctx.lineTo(curX, curY);
                ctx.strokeStyle = "#000000";
                ctx.stroke();
                canvas_data.pencil.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
            }                
        }
        else if ( touchpad_option == false ){
            draw();

            function draw(){
                ctx.lineTo(curX, curY);
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();
                canvas_data.eraser.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
            }
    
        }

    
    }
});

// =================================================================================================
// =================================================================================================

// Add event listener for touchend
canvas.addEventListener("touchend", function(e){
    e.preventDefault();
    hold = false;

    if ( touchpad_option == true ){
        downloadImage();
    }
});

// =================================================================================================
// =================================================================================================

// Add event listener for touchcancel
canvas.addEventListener("touchcancel", function(e){
    e.preventDefault();
    hold = false;
});

// =================================================================================================
// =================================================================================================

// canvas pencil function
function pencil(){
    ctx.globalCompositeOperation = "source-over";
    touchpad_option = true;
    canvas.style.cursor = "url('static/images/pencil.png'), auto";
    canvas.onmousedown = function(e){
        curX = e.clientX - canvas.offsetLeft;
        curY = e.clientY - canvas.offsetTop;
        hold = true;
            
        prevX = curX;
        prevY = curY;
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
    };
        
    canvas.onmousemove = function(e){
        if(hold){
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw();
        }
    };
        
    canvas.onmouseup = function(e){
        hold = false;
        downloadImage();
    };
        
    canvas.onmouseout = function(e){
        hold = false;
    };
        
    function draw(){
        ctx.lineTo(curX, curY);
        ctx.strokeStyle = "#000000";
        ctx.stroke();
        canvas_data.pencil.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
    }
}
   
// =================================================================================================
// =================================================================================================
        
// canvas eraser function
function eraser(){
    ctx.globalCompositeOperation = "destination-out";
    touchpad_option = false;
    canvas.style.cursor = "url('static/images/eraser.png'), auto";
    canvas.onmousedown = function(e){
        curX = e.clientX - canvas.offsetLeft;
        curY = e.clientY - canvas.offsetTop;
        hold = true;
            
        prevX = curX;
        prevY = curY;
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
    };
        
    canvas.onmousemove = function(e){
        if(hold){
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw();
        }
    };
        
    canvas.onmouseup = function(e){
        hold = false;
        downloadImage();
    };
        
    canvas.onmouseout = function(e){
        hold = false;
    };
        
    function draw(){
        ctx.lineTo(curX, curY);
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
        canvas_data.pencil.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
    }    

}  

// =================================================================================================
// =================================================================================================

// for asking user to confirm reset
document.getElementById("canvas_reset").addEventListener("click", function() {
    document.getElementById("confirm-dialog_reset").style.display = "block";
    }
);

document.getElementById("yes-button_reset").addEventListener("click", function() {
    reset();
    document.getElementById("confirm-dialog_reset").style.display = "none";
    } 
);

document.getElementById("no-button_reset").addEventListener("click", function() {
    document.getElementById("confirm-dialog_reset").style.display = "none";
    }
);

// =================================================================================================
// =================================================================================================

function downloadImage() {
    const shadow = document.getElementById("shadow");
    const paint = document.getElementById("paint");
    var data = paint.toDataURL();    
    shadow.style.backgroundImage = 'none';
    $.ajax({
        type: "POST",
        url: "/update_shadow",
        data: {image: data},
        success: function(response) {
          shadow.style.backgroundImage = "url(data:image/png;base64," + response + ")";
        }
    });
}

// =================================================================================================
// =================================================================================================

document.getElementById("option-1").addEventListener("click", function() {
    isMale = true;
    console.log("Check2 : ",isMale);
    shadow.style.backgroundImage = "url('/static/images/male_image.png')";

    $.ajax(
        {
            type: 'POST',
            url: "change_label",
            data: JSON.stringify({isMale: isMale}),
            contentType: 'application/json',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.error(error);
            }
        }
    );

});

// =================================================================================================
// =================================================================================================

document.getElementById("option-2").addEventListener("click", function() {
    isMale = false;
    console.log("Check2 : ",isMale);
    shadow.style.backgroundImage = "url('/static/images/female_image.png')";

    $.ajax(
        {
            type: 'POST',
            url: "change_label",
            data: JSON.stringify({isMale: isMale}),
            contentType: 'application/json',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.error(error);
            }
        }
    );

});

// =================================================================================================
// =================================================================================================

function savefinalimage() {
    var image = document.getElementById("generated_final_image");
    var link = document.createElement('a');
    link.download = "final_image.png";
    link.href = image.src;
    link.click();
}