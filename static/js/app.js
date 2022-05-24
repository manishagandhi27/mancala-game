var game = (function () {
	'use strict';
   document.getElementById("turn").style.display = 'none';
              document.getElementById("winnner").style.display = 'none';
   
	var gameid ="";
	var init_pits = function (player, row, data, start,end) {	
		for (var pit = 0; pit < row.length; pit++) {
			//row[pit].setAttribute('data-pit', pit);
			row[pit].onclick = function(e) {                
                updatePits(e.target.getAttribute("data-pit"));
            };
             //document.querySelectorAll('.player-two')[0].textContent = data.pits[13].stones;
             //document.querySelectorAll('.player-one')[0].textContent = data.pits[6].stones;
              document.querySelectorAll('.player-one store .pit')
            if(player =="two"){
            row[pit].textContent = data.pits[pit+7].stones;
            } else {
			row[pit].textContent = data.pits[pit].stones;
			}
		}
		 document.getElementsByClassName("player-two store")[0].textContent  = data.pits[13].stones;
         document.getElementsByClassName("player-one store")[0].textContent  = data.pits[6].stones;
	};
	
	
  function creategame() {

    var search = {};  
    $.ajax({
        type: "POST",
        contentType: "application/json",
        url: "/games",
        data: JSON.stringify(search),
        dataType: 'json',
        cache: false,
        timeout: 600000,
        success: function (data) {

            var json = "<h4>Ajax Response</h4>&lt;pre&gt;"
                + JSON.stringify(data, null, 4) + "&lt;/pre&gt;";
            $('#feedback').html(json);
           gameid = data.id;
            console.log("SUCCESS : ", data);
            init_pits('one', document.querySelectorAll('.row.player-one .pit'), data);
	        init_pits('two', document.querySelectorAll('.row.player-two .pit'), data);
	           document.getElementsByClassName("current-player")[0].textContent  = data.playerTurn;
	        

        },
        error: function (e) {

            var json = "<h4>Ajax Response</h4>&lt;pre&gt;"
                + e.responseText + "&lt;/pre&gt;";
            $('#feedback').html(json);

            console.log("ERROR : ", e);
            $("#btn-search").prop("disabled", false);

        }
    });

}
var get_other_player = function () {
		return this.player === 'one' ? 'two' : 'one';
	};
function updatePits(pitid) {

    var search = {};  
    $.ajax({
        type: "PUT",
        contentType: "application/json",
        url: "/games/"+gameid +"/pits/"+pitid,
        data: JSON.stringify(search),
        dataType: 'json',
        cache: false,
        timeout: 600000,
        success: function (data) {

            var json = "<h4>Ajax Response</h4>&lt;pre&gt;"
                + JSON.stringify(data, null, 4) + "&lt;/pre&gt;";
            $('#feedback').html(json);

            console.log("SUCCESS : ", data);
            init_pits('one', document.querySelectorAll('.row.player-one .pit'), data);
	        init_pits('two', document.querySelectorAll('.row.player-two .pit'), data);
            document.getElementsByClassName("current-player")[0].textContent  = data.playerTurn;
           document.getElementById("turn").style.display = 'block';
        },
        error: function (e) {

            var json = "<h4>Ajax Response</h4>&lt;pre&gt;"
                + e.responseText + "&lt;/pre&gt;";
            $('#feedback').html(json);

            console.log("ERROR : ", e);
            $("#btn-search").prop("disabled", false);

        }
    });

}
	
	document.querySelector('.new-game').onclick = function () {
		creategame();
		
	};
	var switch_turn = function () {
		this.player = this.get_other_player();
		

		var player = this.player;
		setTimeout(function () {
			document.body.setAttribute('data-player', player);
			document.querySelector('.current-player').textContent = player;
		}, 700 );
	};
	
	
	return game;
})();
