console.log('script running')

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
            picFrame.image = e.target.result

            data = (e.target.result).split(',')[1]

            axios.get('/api/data:' + data)
            .then(function (response) {
              console.log('response was:', response);
              picFrame.message = "That is a picture of the digit " + response.data.guess
            })
        };

        reader.readAsDataURL(input.files[0]);
    }
}
