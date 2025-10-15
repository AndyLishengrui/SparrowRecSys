

 function appendMovie2Row(rowId, movieName, movieId, year, rating, rateNumber, genres, baseUrl) {

    var genresStr = "";
    $.each(genres, function(i, genre){
        genresStr += ('<div class="genre"><a href="'+baseUrl+'collection.html?type=genre&value='+genre+'"><b>'+genre+'</b></a></div>');
    });


    var divstr = '<div class="movie-row-item" style="margin-right:5px">\
                    <movie-card-smart>\
                     <movie-card-md1>\
                      <div class="movie-card-md1">\
                       <div class="card">\
                        <link-or-emit>\
                         <a uisref="base.movie" href="./movie.html?movieId='+movieId+'">\
                         <span>\
                           <div class="poster">\
                            <img src="./posters/' + movieId + '.jpg" />\
                           </div>\
                           </span>\
                           </a>\
                        </link-or-emit>\
                        <div class="overlay">\
                         <div class="above-fold">\
                          <link-or-emit>\
                           <a uisref="base.movie" href="./movie.html?movieId='+movieId+'">\
                           <span><p class="title">' + movieName + '</p></span></a>\
                          </link-or-emit>\
                          <div class="rating-indicator">\
                           <ml4-rating-or-prediction>\
                            <div class="rating-or-prediction predicted">\
                             <svg xmlns:xlink="http://www.w3.org/1999/xlink" class="star-icon" height="14px" version="1.1" viewbox="0 0 14 14" width="14px" xmlns="http://www.w3.org/2000/svg">\
                              <defs></defs>\
                              <polygon fill-rule="evenodd" points="13.7714286 5.4939887 9.22142857 4.89188383 7.27142857 0.790044361 5.32142857 4.89188383 0.771428571 5.4939887 4.11428571 8.56096041 3.25071429 13.0202996 7.27142857 10.8282616 11.2921429 13.0202996 10.4285714 8.56096041" stroke="none"></polygon>\
                             </svg>\
                             <div class="rating-value">\
                              '+rating+'\
                             </div>\
                            </div>\
                           </ml4-rating-or-prediction>\
                          </div>\
                          <p class="year">'+year+'</p>\
                         </div>\
                         <div class="below-fold">\
                          <div class="genre-list">\
                           '+genresStr+'\
                          </div>\
                          <div class="ratings-display">\
                           <div class="rating-average">\
                            <span class="rating-large">'+rating+'</span>\
                            <span class="rating-total">/5</span>\
                            <p class="rating-caption"> '+rateNumber+' ratings </p>\
                           </div>\
                          </div>\
                         </div>\
                        </div>\
                       </div>\
                      </div>\
                     </movie-card-md1>\
                    </movie-card-smart>\
                   </div>';
    $('#'+rowId).append(divstr);
};


function addRowFrame(pageId, rowName, rowId, baseUrl) {
 var divstr = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 <a class="plainlink" title="go to the full list" href="'+baseUrl+'collection.html?type=genre&value='+rowName+'">' + rowName + '</a> \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + rowId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
     $(pageId).prepend(divstr);
};

function addRowFrameWithoutLink(pageId, rowName, rowId, baseUrl) {
 var divstr = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 <a class="plainlink" title="go to the full list" href="'+baseUrl+'collection.html?type=genre&value='+rowName+'">' + rowName + '</a> \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + rowId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
     $(pageId).prepend(divstr);
};

function addGenreRow(pageId, rowName, rowId, size, baseUrl) {
    addRowFrame(pageId, rowName, rowId, baseUrl);
    $.getJSON(baseUrl + "getrecommendation?genre="+rowName+"&size="+size+"&sortby=rating", function(result){
        $.each(result, function(i, movie){
          appendMovie2Row(rowId, movie.title, movie.movieId, movie.releaseYear, movie.averageRating.toPrecision(2), movie.ratingNumber, movie.genres,baseUrl);
        });
    });
};

function addRelatedMovies(pageId, containerId, movieId, baseUrl){

    var rowDiv = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 Related Movies \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + containerId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
    $(pageId).prepend(rowDiv);

    $.getJSON(baseUrl + "getsimilarmovie?movieId="+movieId+"&size=16&model=emb", function(result){
            $.each(result, function(i, movie){
              appendMovie2Row(containerId, movie.title, movie.movieId, movie.releaseYear, movie.averageRating.toPrecision(2), movie.ratingNumber, movie.genres,baseUrl);
            });
    });
}

function addUserHistory(pageId, containerId, userId, baseUrl){

    var rowDiv = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 User Watched Movies \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + containerId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
    $(pageId).prepend(rowDiv);

    $.getJSON(baseUrl + "getuser?id="+userId, function(userObject){
            $.each(userObject.ratings, function(i, rating){
                $.getJSON(baseUrl + "getmovie?id="+rating.rating.movieId, function(movieObject){
                    appendMovie2Row(containerId, movieObject.title, movieObject.movieId, movieObject.releaseYear, rating.rating.score, movieObject.ratingNumber, movieObject.genres, baseUrl);
                });
            });
    });
}

function addRecForYou(pageId, containerId, userId, model, baseUrl){

    console.log("addRecForYou called with:", pageId, containerId, userId, model, baseUrl);

    // 根据模型类型设置不同的标题
    var sectionTitle = "Recommended For You";
    if (model === "popularity") {
        sectionTitle = "热度推荐 (Most Popular Movies)";
    } else if (model === "emb") {
        sectionTitle = "基于嵌入向量推荐";
    } else if (model === "nerualcf") {
        sectionTitle = "深度学习推荐";
    }

    console.log("Section title:", sectionTitle);

    var rowDiv = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 ' + sectionTitle + ' \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + containerId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
    
    console.log("Adding row div to:", pageId);
    $(pageId).prepend(rowDiv);

    var apiUrl = baseUrl + "getrecforyou?id="+userId+"&size=32&model=" + model;
    console.log("Making API call to:", apiUrl);

    $.getJSON(apiUrl, function(result){
                console.log("API response received:", result);
                $.each(result, function(i, movie){
                  console.log("Processing movie:", movie.title);
                  appendMovie2Row(containerId, movie.title, movie.movieId, movie.releaseYear, movie.averageRating.toPrecision(2), movie.ratingNumber, movie.genres,baseUrl);
                });
     }).fail(function(jqXHR, textStatus, errorThrown) {
         console.error("API call failed:", textStatus, errorThrown);
         console.error("Response:", jqXHR.responseText);
     });
}


function addMovieDetails(containerId, movieId, baseUrl) {

    $.getJSON(baseUrl + "getmovie?id="+movieId, function(movieObject){
        var genres = "";
        $.each(movieObject.genres, function(i, genre){
                genres += ('<span><a href="'+baseUrl+'collection.html?type=genre&value='+genre+'"><b>'+genre+'</b></a>');
                if(i < movieObject.genres.length-1){
                    genres+=", </span>";
                }else{
                    genres+="</span>";
                }
        });

        var ratingUsers = "";
                $.each(movieObject.topRatings, function(i, rating){
                        ratingUsers += ('<span><a href="'+baseUrl+'user.html?id='+rating.rating.userId+'"><b>User'+rating.rating.userId+'</b></a>');
                        if(i < movieObject.topRatings.length-1){
                            ratingUsers+=", </span>";
                        }else{
                            ratingUsers+="</span>";
                        }
                });

        var movieDetails = '<div class="row movie-details-header movie-details-block">\
                                        <div class="col-md-2 header-backdrop">\
                                            <img alt="movie backdrop image" height="250" src="./posters/'+movieObject.movieId+'.jpg">\
                                        </div>\
                                        <div class="col-md-9"><h1 class="movie-title"> '+movieObject.title+' </h1>\
                                            <div class="row movie-highlights">\
                                                <div class="col-md-2">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Release Year</div>\
                                                        <div> '+movieObject.releaseYear+' </div>\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Links</div>\
                                                        <a target="_blank" href="http://www.imdb.com/title/tt'+movieObject.imdbId+'">imdb</a>,\
                                                        <span><a target="_blank" href="http://www.themoviedb.org/movie/'+movieObject.tmdbId+'">tmdb</a></span>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-3">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> MovieLens predicts for you</div>\
                                                        <div> 5.0 stars</div>\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> Average of '+movieObject.ratingNumber+' ratings</div>\
                                                        <div> '+movieObject.averageRating.toPrecision(2)+' stars\
                                                        </div>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-6">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Genres</div>\
                                                        '+genres+'\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Who likes the movie most</div>\
                                                        '+ratingUsers+'\
                                                    </div>\
                                                </div>\
                                            </div>\
                                        </div>\
                                    </div>'
        $("#"+containerId).prepend(movieDetails);
    });
};

function addUserDetails(containerId, userId, baseUrl) {

    $.getJSON(baseUrl + "getuser?id="+userId, function(userObject){
        var userDetails = '<div class="row movie-details-header movie-details-block">\
                                        <div class="col-md-2 header-backdrop">\
                                            <img alt="movie backdrop image" height="200" src="./images/avatar/'+userObject.userId%10+'.png">\
                                        </div>\
                                        <div class="col-md-9"><h1 class="movie-title"> User'+userObject.userId+' </h1>\
                                            <div class="row movie-highlights">\
                                                <div class="col-md-2">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">#Watched Movies</div>\
                                                        <div> '+userObject.ratingCount+' </div>\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> Average Rating Score</div>\
                                                        <div> '+userObject.averageRating.toPrecision(2)+' stars\
                                                        </div>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-3">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> Highest Rating Score</div>\
                                                        <div> '+userObject.highestRating+' stars</div>\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> Lowest Rating Score</div>\
                                                        <div> '+userObject.lowestRating+' stars\
                                                        </div>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-6">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Favourite Genres</div>\
                                                        '+'action'+'\
                                                    </div>\
                                                </div>\
                                            </div>\
                                        </div>\
                                    </div>'
        $("#"+containerId).prepend(userDetails);
    });
};

// ========== 模型管理功能 ==========

// 加载可用的模型列表
function loadModelList() {
    $.get("/getmodel?action=list", function(data) {
        var modelSelect = $("#modelSelect");
        modelSelect.empty();
        
        if (data.success && data.models) {
            $.each(data.models, function(i, model) {
                var option = $('<option></option>')
                    .attr('value', model.version)
                    .text(model.displayName);
                
                if (model.isCurrent) {
                    option.attr('selected', 'selected');
                    $("#modelStatus").text("当前模型: " + model.displayName).css('color', 'green');
                }
                
                modelSelect.append(option);
            });
        } else {
            modelSelect.append('<option value="">加载失败</option>');
            $("#modelStatus").text("加载模型列表失败").css('color', 'red');
        }
    }).fail(function() {
        $("#modelSelect").html('<option value="">服务器错误</option>');
        $("#modelStatus").text("无法连接到服务器").css('color', 'red');
    });
}

// 切换模型
function switchModel() {
    var selectedVersion = $("#modelSelect").val();
    if (!selectedVersion) {
        alert("请选择一个模型版本");
        return;
    }
    
    // 显示加载状态
    $("#modelStatus").text("正在切换模型...").css('color', 'orange');
    $("#switchModelBtn").prop('disabled', true).text('切换中...');
    
    $.post("/getmodel", {
        action: "switch",
        version: selectedVersion
    }, function(data) {
        if (data.success) {
            $("#modelStatus").text("✅ " + data.message).css('color', 'green');
            
            // 刷新推荐结果
            refreshRecommendations();
            
        } else {
            $("#modelStatus").text("❌ " + data.message).css('color', 'red');
        }
    }).fail(function() {
        $("#modelStatus").text("❌ 切换失败：服务器错误").css('color', 'red');
    }).always(function() {
        $("#switchModelBtn").prop('disabled', false).text('切换模型');
    });
}

// 刷新推荐结果
function refreshRecommendations() {
    // 清空现有推荐结果
    $("#for-you-rec").empty();
    $("#similar-movie-rec").empty();
    $("#other-rec").empty();
    
    // 重新加载推荐
    loadDefaultRecList();
    
    alert("模型切换成功！推荐结果已更新。");
}

// 页面加载完成后初始化模型管理
$(document).ready(function() {
    // 加载模型列表
    loadModelList();
    
    // 绑定切换模型按钮事件
    $("#switchModelBtn").click(function() {
        switchModel();
    });
});





