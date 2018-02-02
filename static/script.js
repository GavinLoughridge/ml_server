Vue.options.delimiters = ['{[{', '}]}'];

let digitReader = new Vue({
  el: '#digitReader',
  data: {
    original: '/static/assets/upload.jpg',
    processed: '/static/assets/upload.jpg',
    guess: '?'
  }
})

function capture() {
  $('#pictureInput').trigger("click")
}

function upload(input) {
  if (input.files && input.files[0]) {
    loadImage(
      input.files[0],
      function (rotated_canvas) {
        rotated_canvas.toBlob(function(blob) {
          let reader = new FileReader();

          reader.onload = function (e) {
              digitReader.original = e.target.result

              file = (e.target.result).split(',')
              type = file[0]
              content = file[1]

              axios.post('/api', {
                data: content
              })
              .then(function (response) {
                digitReader.guess = response.data.guess

                digitReader.processed = type + ',' + response.data.image
              })
          }

          reader.readAsDataURL(blob);
        })
      },
      {
        orientation: true
      }
    );
  }
}
