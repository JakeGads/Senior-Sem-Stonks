console.log('director reporting for duty');

document.getElementById('btn').onclick = function() {
    let tags = document.getElementById('tagTB').value;
    let start_date = document.getElementById('startDate').value;
    
    console.log(tags);

    // if (tags=""){
    //     location.href = "/";
    // }

    // if(start_date !== ""){
    //     location.href = "index/" + tags + "/" + start_date;
    // }
    // else if (start_date === ""){
    //     location.href = "index/" + tags;
    // }   
}