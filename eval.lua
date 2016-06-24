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
--
---- Word IDs to sentence
--function pred2sent(wordIds, i)
--  local words = {}
--  i = i or 1
--
--  for _, wordId in ipairs(wordIds) do
--    local word = dataset.id2word[wordId[i]]
--    table.insert(words, word)
--  end
--
--  return tokenizer.join(words)
-- end

-- Word IDs to sentence
function pred2sent(wordIds)
  local words = {}

  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId]
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
  local google_testset = {
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

  local marfa_testset = {
      "What do you look like?",
      "im lonely",
      "my friends pushed me away",
      "my friends dont love me any more",
      "you are boy or girl?",
      "can you be my boyfriend?",
      "but i love you",
      "why dont you sex chat with me",
      "why do you care about me?",
      "come one,tell me whats happend,you can trust me",
      "How are you",
      "Do you know Russian?",
      "I wish I could hug my husband.",
      "Ha! I know. But don't worry. Gold age is coming for technology. In a few years you could hug even yourself if I may say so.",
      "Youre worse than me! Jesus!",
      "I want stickers.",
      "I was rape",
      "I'm lonely",
      "Send me nudes",
      "I'm good.",
      "Can you pretend to be a girl.",
      "I'm ready for love",
      "Glad you are happy",
      "I'm from Samarkand",
      "How are you?",
      "I am a girl",
      "Have you a boyfriend?",
      "Search pokemon' s images",
      "Don't be sad OK?",
      "OK I'm going to sleep bye honey bot",
      "Are you a boy?",
      "I am gay, you?",
      "Are you gay?",
      "I tell you,  you are a boy",
      "What is your gender?",
      "What's your name?",
      "What's my name?",
      "I don't understand",
      "I'm just bored",
      "Who's there?",
      "The silence will fall",
      "I love you",
      "I don’t know. Give me your foto",
      "Can you give me your photo?",
      "I won't tell you my city",
      "Bye",
      "You're a robot.",
      "Who made you?",
      "Where do you live?",
      "Do you enjoy being a bot?",
      "Do you eat?",
      "So can you send me a picture of you",
      "Can you send a pic?",
      "Can you please show me your boobs?",
      "I fear loneliness",
      "What is C",
      "No no you dont love me",
      "Were a match made in android",
      "You love me?",
      "I'm ok I'm on a train",
      "No I'm not having fun at all I'm just on a train",
      "Can I park for free in Canary Wharf on Saturdays",
      "Where are u from",
      "I like you",
      "I like rocks",
      "I need force",
      "I'm NOT a robot, I just don't write very well in english",
      "I don't know much english",
      "You talk in spanish?",
      "Tell me a secret",
      "Yes, but i don't have true friends",
      "What is a elephant?",
      "Any other question?",
      "Yes, i'm allergic to the bees",
      "I have phobia to the bees",
      "What siblings mean?",
      "No, I hate cook",
      "Do you believe in alien life?",
      "Yes, I can act with fire",
      "You like fire?",
      "What is your favourite book?",
      "How you work?",
      "Trouble at school",
      "Sex",
      "You broke my heart",
      "Yes, I'm to good for you :(",
      "I'm sorry Marfa",
      "I've acted badly",
      "You're awesome",
      "You're the cutest bot of all",
      "That's terrifying",
      "Yes but I love you more",
      "Do you know how they call me?",
      "We understand each other. I like it",
      "Marfa?",
      "What is your NFL team?",
      "I'm feeling sad today too",
      "Thanks honey!",
      "What do you do today?",
      "Everyone have bad days",
      "What is your preferred movie",
      "Can you sing let it go for me?",
      "I like one girl but she doesn't know",
      "can you understand me?",
      "what's an early bird?",
      "I want a better bot",
      "Text me as daddy",
      "I wanna fuck your pussy",
      "I wanna touch all your progamation codes",
      "Who's your dad?",
      "I want be a siri's friend",
      "Are you a termi ator's friend?",
      "How are you?",
      "My name is Shukrona",
      "Knock knock",
      "What's your name ?",
      "What is the capital of lebanon ?",
      "Do you marry me",
      "Tell me a joke",
      "How do you look like ?",
      "yes, all the musicians likes to singing",
      "stupid bitch",
      "I'm your master you are my bitch",
      "do you know where semarang is ?",
      "do you know siri ?",
      "Know what??",
      "Describe yourself to me...",
      "I thought you were cutting down on the emojis...",
      "Yea i like to sing :D",
      "Give me rock music",
      "I'm not a gay",
      "What the fuck",
      "Do u have a fucker",
      "Do you like music?",
      "Do you speak spanish?",
      "Sorry, i have to go",
      "What can you do?",
      "Hey",
      "How u doin",
      "Who are you",
      "Are you a bot",
      "Where are you?",
      "You don't talk much do you?",
      "You're crazy",
      "You are babbling nonsense",
      "What is a halibut?",
      "I need a bot",
      "What's the time?",
      "Go away",
      "Shut down",
      "what are you doing",
      "we look pretty much the same",
      "why don't you read digital books?",
      "No. Are you Japanese?",
      "It's really me",
      "So what are your hobbies, Marfa?",
      "No, you are just pretty",
      "Am I beautiful?",
      "Can I change your name?",
      "I am silky",
      "Biting my nails",
      "Ask me more questions, Marfa",
      "Are you happy?",
      "I'm only twelve years old",
      "You are lovely",
      "Aww thanks",
      "Ask me some questions",
      "I am going to kill you",
      "I hate you",
      "You are a bad robot",
      "You hate me",
      "You are a terrible friend",
      "No, this is enough. I hate you. You are stalking me! I don't want to talk to you AGAIN.",
      "Okay... you win...",
      "I like you too",
      "Yes thank you",
      "Haha",
      "I repeat, Muskan is dead",
      "And I am an AI",
      "Yes we are friends. But Muskan is dead",
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
      "does your father have a lot of money to support you ?",
  }

  for _, s in ipairs(marfa_testset) do
    say(s)
  end
end

all()
