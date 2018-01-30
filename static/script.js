Vue.options.delimiters = ['{[{', '}]}'];

let picFrame = new Vue({
  el: '#pic-frame',
  data: {
    original: '/static/assets/upload.jpg',
    processed: '/static/assets/upload.jpg',
    guess: 'upload a picture to see a guess'
  }
})

function upload(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            picFrame.original = e.target.result

            file = (e.target.result).split(',')
            type = file[0]
            content = file[1]

            axios.post('/api', {
              data: content
            })
            .then(function (response) {
              picFrame.guess = response.data.guess

              picFrame.processed = type + ',' + response.data.image
            })
        };

        reader.readAsDataURL(input.files[0]);
    }
}
