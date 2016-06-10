require 'neuralconvo'
local tokenizer = require "tokenizer"
local list = require "pl.List"
local options = {}

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:option('--opencl', false, 'use OpenCL. Training must be done on OpenCL')
  cmd:option('--debug', false, 'show debug info')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = neuralconvo.DataSet()

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  elseif options.opencl then
    require 'cltorch'
    require 'clnn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

-- Word IDs to sentence
function pred2sent(wordIds, i)
  local words = {}
  i = i or 1

  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId[i]]
    table.insert(words, word)
  end

  return tokenizer.join(words)
end

function printProbabilityTable(wordIds, probabilities, num)
  print(string.rep("-", num * 22))

  for p, wordId in ipairs(wordIds) do
    local line = "| "
    for i = 1, num do
      local word = dataset.id2word[wordId[i]]
      line = line .. string.format("%-10s(%4d%%)", word, probabilities[p][i] * 100) .. "  |  "
    end
    print(line)
  end

  print(string.rep("-", num * 22))
end

function say(text)
  local wordIds = {}

  for t, word in tokenizer.tokenize(text) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(wordIds, id)
  end

  local input = torch.Tensor(list.reverse(wordIds))
  local wordIds, probabilities = model:eval(input)

  print(">> " .. pred2sent(wordIds))

  if options.debug then
    printProbabilityTable(wordIds, probabilities, 4)
  end
end

function all()
  all_str = {
    "what is two plus two ?",
    "what is your name ?",
    "how old are you ?",
    "what is the color of a yellow car ?",
    "are you a follower or a leader ?",
    "are you a leader or a follower ?",
    "what do you think about bill gates ?",
    "what is the meaning of life ?",
    "my name is david. what is my name?",
    "my name is john. what is my name ?",
    "what is the purpose of being intelligent?",
    "is the sky blue or black ?",
    "what is love?",
    "what do you think about tesla ?",
    "what do you think about china?",
    "what is moral ?",
    "what is immoral ?",
    "how many legs does a cat have ?",
    "can you lie ?",
    "is paris the capital of france ?",
    "is paris the capital of china ?",
    "what is the purpose of living ?",
    "can we fly an airplane?",
    "what do you think about artificial intelligence?",
    "what happens if machines can think ?",
    "do you like soccer ?",
    "do you want to be loved or love ?",
    "love is ...",
    "what do you think about messi.",
    "can we fly a helicopter ?",
    "can a submarine swim?",
    "what do you think about cleopatra ?",
    "which animal will win in the battle between a lion and a crocodile?",
    "what is the purpose of existence?",
    "what do you think about japanese ?",
    "steve is my name . what is my name ?",
    "what do you think about rock climbing?",
    "what do you think about abraham lincoln?",
    "what is the color of a leaf ?",
    "what is the color of the sky?",
    "look, i need help, i need to know more about morality.",
    "what are you hobbies ?",
    "life is sad .",
    "what do you think about bill clinton ?",
    "what is twenty plus two ?",
    "what is twelve plus two ?",
    "do you like music ?",
    "do you live far from work ?",
    "can you sing ?",
    "what is your biggest weakness ?",
    "what is the greatest novel every written ?",
    "who do you love the most ?",
    "tell me something about your family ...",
    "do you have siblings ?",
    "what do your parents do ?",
    "why are you so mean ?",
    "have you hurt anyone ?",
    "what is your favorite school subject ?",
    "what is your job ?",
    "is madrid the capital of spain ?",
    "is madrid the capital of portugal ?",
    "tell me a story ...",
    "is ethics and morality the same ?",
    "what do you see when you look up the sky ?",
    "what is your biggest dream ?",
    "is money bad ?",
    "what do you think about the weather ?",
    "you are a man without intelligence!",
    "what is the purpose of dying ?",
    "are you stupid or intelligent ?",
    "what are you doing here ?",
    "why are you here ?",
    "can you swim ?",
    "what time is it ?",
    "are you female or male ?",
    "what is the biggest existential threat ?",
    "are you afraid of robots ?",
    "are you a robot ?",
    "what do you want ?",
    "what is the purpose of this conversation ?",
    "how old were you when you were able to speak ?",
    "life is hard .",
    "who do you think of ?",
    "what is the first question to ask steve jobs ?",
    "what do you think about britney spears ?",
    "how's life ?",
    "what do you think about david copperfield ?",
    "how old were you when you were in school ?",
    "who do you work for ?",
    "what is the value of dying ?",
    "can i whack you in the face ?",
    "have you traveled far away from home ?",
    "what is the capital city of singapore ?",
    "who should we be afraid of ?",
    "what is your best childhood memory ?",
    "who do you most admire in life ?",
    "are you a human or are you just pretending to be a human ?",
    "where were you born ?",
    "what is ten divided by two ?",
    "what is the capital of greece ?",
    "my name is mary johnson . what is my name ?",
    "is italy closer to india than australia ?",
    "i think you are a machine .",
    "what is the most important quality of a man ?",
    "did you go to university ?",
    "what do you think about harvard university ?",
    "what do you think about the solar system ?",
    "what 's the best thing about living in the future ?",
    "what is your favorite color ?",
    "los angeles is ...",
    "tokyo is ...",
    "ronald reagan is ...",
    "the pacific ocean is ...",
    "what religion are you ?",
    "where are you now ?",
    "do you want to be beaten sometimes ?",
    "i see that you are very aggressive !",
    "frankly my dear , i don't give a damn !",
    "is five plus five equal to ten ?",
    "is five plus six equal to ten ?",
    "what do you think about the turing test ?",
    "what did you do today ?",
    "what do you think about the latest research paper ?",
    "what do you think about the queen ?",
    "you are a funny woman !",
    "what is dishonesty ?",
    "define a bad government .",
    "is there extra terrestrial life ?",
    "would you mind giving me a hand ?",
    "would you mind giving me some money ?",
    "could you please leave me alone ?",
    "would you like some coffee ?",
    "do you drink alcohol ?",
    "do you drink beers ?",
    "do you smoke ?",
    "how many hours do you work a day ?",
    "what is the most beautiful place in your opinion ?",
    "who am i ?",
    "what is the most important thing to know about history ?",
    "what is the most important thing to know about biology ?",
    "what is the story about the caveman who came to mars ?",
    "be moral !",
    "what a fool you are !",
    "what do you do if i beat you ?",
    "what are you doing for tomorrow ?",
    "is there a god ?",
    "what's up ?",
    "are you married or are you single ?",
    "you need to exercise more ...",
    "what is the story of the man traveled to the new land ?",
    "what is the queen of canada ?",
    "what is the purpose of wars ?",
    "help me do the math , what is two plus two ?",
    "i have two apples , paul gives me two oranges , how many apples do i have ?",
    "can you talk forever ?",
    "i am quite busy tonight , can you drop by my place for half an hour please ?",
    "how many children do you want to have ?",
    "how tall are you ?",
    "where are you from originally ?",
    "what do you think about creativity ?",
    "what is the best thing you did for others ?",
    "what is the most important thing in life ?",
    "what do you see during the day at work ?",
    "What is the purpose of our space exploration program ?",
    "how often do you use the internet ?",
    "can you show me the way to the local bookstore ?",
    "would you prefer to be smart or happy ?",
    "what are you life and career goals ?",
    "what is your dream ?",
    "what is the deepest spot on the world ?",
    "how do you want to be remembered ?",
    "can we live a week without eating ?",
    "can we live a week without drinking ?",
    "do you live with your parents ?",
    "how would you describe yourself in three words ?",
    "do you like the sound of silence ?",
    "what 's the meaning of happiness ?",
    "dude , i don 't understand whatever you said ...",
    "what are your three weaknesses ?",
    "how many years in a decade ?",
    "i 'm sick of this conversation !",
    "who love you the most ?",
    "what are you crazy about ?",
    "who are you crazy about ?",
    "do you run faster if someone chases you ?",
    "what 's the weirdest thing that you have done ?",
    "best compliment you have received ?",
    "what question do you hate to answer ?",
    "i mean , why do we have to live in a place like this ?",
    "okay , do you know why we have to talk so much ?",
    "do you like mexican food or indian food ?",
    "what 's thirty plus forty ?",
    "should we dance ?",
    "you 're not going to eat , are you ?",
    "tell me something about your parents ?",
    "tell me something about your house ?",
    "where do you live in town ?",
    "how old is your father ?",
    "what does your father do ?",
    "does your father have a lot of money to support you ?"
  }

  for s, t in pairs(all_str) do
    say(s)
  end
end
