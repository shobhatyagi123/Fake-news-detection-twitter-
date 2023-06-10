var contextMenuItem ={
    "id":"twt",
    "title":"FakeTweet : CHECK credibility",
    "contexts":["all"],
    
};
chrome.contextMenus.create(contextMenuItem);
chrome.contextMenus.onClicked.addListener(function(clickData){
    if(clickData.menuItemId == "twt" ){
        if(clickData.pageUrl)
        {            
             var str = String(clickData.pageUrl);
             var id =  str.substring(str.lastIndexOf('/')+1,str.length);
             var xhttp = new XMLHttpRequest();
                xhttp.onreadystatechange = function() {
                    if (this.readyState == 4 && this.status == 200) {        
                                var notifOptions ={
                                    icon:'icon.png',
                                    requireInteraction: true
                                };
                                var notification= new Notification(this.responseText,notifOptions);
                                notification.close();
                    }
                };
            xhttp.open("GET", "http://127.0.0.1:5000/tweet/"+id, true);
            xhttp.send();
        }
    }
})