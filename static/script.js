Vue.options.delimiters = ['{[{', '}]}'];

let picFrame = new Vue({
  el: '#pic-frame',
  data: {
    image: '/static/assets/upload.jpg',
    message: 'upload a picture to see a guess'
  }
})

function upload(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            // picFrame.image = e.target.result

            file = (e.target.result).split(',')
            type = file[0]
            content = file[1]

            axios.post('/api', {
              data: content
            })
            .then(function (response) {
              picFrame.message = "That is a picture of the digit " + response.data.guess

              picFrame.image = type + ',' + response.data.image
            })
        };

        reader.readAsDataURL(input.files[0]);
    }
}
