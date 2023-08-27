function ImgSwitch(){
    img_dom = document.getElementById("overlaid_img");
    if(img_dom.style.display === ""){ // first time & default
        img_dom.style.display = "initial";
    }
    else if(img_dom.style.display === "initial"){
        img_dom.style.display = "";
    }

    slider = document.getElementById("slider");
    if(slider.style.display === ""){ // first time & default
        slider.style.display = "initial";
        slider.addEventListener("change", function() {
            img_dom.style.opacity = this.value / this.max;
        });
    }
    else if(slider.style.display === "initial"){
        slider.style.display = "";
    }
    
}