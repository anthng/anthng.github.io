$(function () {
    /* NOTE: hard-refresh the browser once you've updated this */
    $("#typed").typed({
      strings: [
        "anthng.profile<br/>" +
        "><span class='caret'>$</span> <mark>FIELDS_OF_INTEREST</mark><br/> ^100" +
        "><span class='caret'>...</span> Natural Language Processing<br/> ^100" +
        "><span class='caret'>...</span> Spatial Temporal Mining<br/> ^100" +
        "><span class='caret'>...</span> Computer Vision<br/> ^100" +
        "><span class='caret'>...</span> Data Mining<br/> ^100" +
        "><span class='caret'>$</span> <mark>TECHINICAL_SKILLS</mark><br/> ^100" +
        "><span class='caret'>...</span> Programming: Python, C#, Java, C++<br/> ^100" +
        "><span class='caret'>...</span> Frameworks: Keras, Tensorflow, Pytorch, Flask<br/> ^100" +
        "><span class='caret'>...</span> Libraries: Sklearn, Pandas, OpenCV, ...<br/> ^100" +
        "><span class='caret'>...</span> Others: Git, Linux<br/> ^100" +
        "><span class='caret'>$</span> <mark>CONTACT</mark> <a style='color : #7ecdcb' href='https://www.linkedin.com/in/anthng/'>Linkedin</a>, <a style='color : #7ecdcb' href='https://github.com/anthng'>Github</a><br/>"
      ],
      showCursor: true,
      cursorChar: '|',
      autoInsertCss: true,
      typeSpeed: 0.0001,
      startDelay: 50,
      loop: false,
      showCursor: false,
      onStart: $('.message form').hide(),
      onStop: $('.message form').show(),
      onTypingResumed: $('.message form').hide(),
      onTypingPaused: $('.message form').show(),
      onComplete: $('.message form').show(),
      onStringTyped: function (pos, self) { $('.message form').show(); },
    });
    $('.message form').hide()
  
  });
  