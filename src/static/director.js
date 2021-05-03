console.log('director reporting for duty');

document.getElementById('btn').onclick = function() {
    let tags = document.getElementById('tagTB').value;
    let start_date = document.getElementById('startDate').value;
    
    console.log(tags);
    tags = tags.replace('$', '');

    let replacements = [',', '\\', '/', '.', ' ', '\t'];

    replacements.forEach(function(i) {
        while (tags.indexOf(i) != -1){
            tags = tags.replace(i, '|');
        }
    });

    console.log(replacements);

    if (tags=""){
        console.log('/')
        window.location = "/";
    }

    if(start_date !== ""){
        console.log("home/" + tags + "/" + start_date)
        window.location = "home/" + tags + "/" + start_date;
    }
    else if (start_date === ""){
        console.log("home/" + tags + "/" + start_date)
        window.location = "home/" + tags;
    }   
}