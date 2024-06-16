let questions = [];
let answers = [];
let candidateMapping = {};
let userBadges = [];
let correctAnswers = 0;
let incorrectAnswers = 0;

console.log("Script loaded successfully");

function displayBadges() {
    const leftList = document.getElementById('earnedBadgesListLeft');
    const rightList = document.getElementById('earnedBadgesListRight');
    if (leftList && rightList) {
        leftList.innerHTML = '';
        rightList.innerHTML = '';
        userBadges.forEach((badgeId, index) => {
            const badge = badges.find(b => b.id === badgeId);
            const li = document.createElement('li');
            li.innerText = badge.name;
            if (index % 2 === 0) {
                leftList.appendChild(li);
            } else {
                rightList.appendChild(li);
            }
        });
    }
}
async function initializeTrivia() {
    try {
        await displayTriviaQuestion();
    } catch (error) {
        console.error('Failed to initialize trivia:', error);
    }
}

async function generateTriviaQuestion() {
    const apiKey = 'sk-proj-PhHFx7jYed8h4763Vbm9T3BlbkFJIHQ3ZEZlB9NoU9vShu2B';
    const apiUrl = 'https://api.openai.com/v1/chat/completions';

    const prompt = "Generate a trivia question and answer about US Presidents, make these unique and/or fun questions, do not repeat questions in one hour. Provide the question in the format 'Question: ...' and the answer in the format 'Answer: ...' with the answer being 'trump' or 'biden'.";

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4o',
                messages: [{ role: 'user', content: prompt }],
                max_tokens: 100
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Error: ${response.status} ${response.statusText} - ${errorData.error.message}`);
        }

        const data = await response.json();
        const messageContent = data.choices[0].message.content.trim();

        const questionMatch = messageContent.match(/Question:\s*(.*?)(?=\s*Answer:)/i);
        const answerMatch = messageContent.match(/Answer:\s*(trump|biden)/i);

        if (questionMatch && answerMatch) {
            return { question: questionMatch[1].trim(), answer: answerMatch[1].trim() };
        } else {
            throw new Error('Failed to parse the generated trivia question and answer.');
        }
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

async function getRandomTriviaQuestion() {
    try {
        const trivia = await generateTriviaQuestion();
        return trivia;
    } catch (error) {
        console.error('Failed to generate trivia question:', error);
        return { question: 'Generating Question, Please Wait', answer: 'trump' };
    }
}

async function displayTriviaQuestion() {
    try {
        const trivia = await getRandomTriviaQuestion();
        document.getElementById('triviaQuestion').innerText = trivia.question;
        document.getElementById('triviaQuestion').dataset.answer = trivia.answer;
        document.getElementById('triviaFeedback').innerText = '';
    } catch (error) {
        console.error('Failed to generate trivia question:', error);
    }
}

function submitAnswer(answer) {
    const triviaQuestionElement = document.getElementById('triviaQuestion');
    const correctAnswer = triviaQuestionElement.dataset.answer;

    if (!correctAnswer) {
        console.error('Correct answer not found');
        document.getElementById('triviaFeedback').innerText = 'Error: Correct answer not found';
        return;
    }

    if (correctAnswer.toLowerCase() === answer.toLowerCase()) {
        correctAnswers++;
        document.getElementById('triviaFeedback').innerText = 'Correct!';
    } else {
        incorrectAnswers++;
        document.getElementById('triviaFeedback').innerText = 'Incorrect!';
    }

    document.getElementById('correctAnswers').innerText = correctAnswers;
    document.getElementById('incorrectAnswers').innerText = incorrectAnswers;

    if (correctAnswers === 1) awardBadge(16);
    if (correctAnswers === 10) awardBadge(17);
    if (correctAnswers === 20) awardBadge(18);

    displayTriviaQuestion();
}

const exampleQuestions = [
    "What is your stance on climate change?",
    "How do you plan to address the healthcare crisis?",
    "What are your thoughts on immigration reform?",
    "How would you handle the current economic situation?",
    "What is your approach to education reform?",
    "What measures would you take to ensure national security?",
    "How do you plan to tackle income inequality?",
    "What are your views on gun control?",
    "How would you address the issue of police reform?",
    "What are your plans for infrastructure development?",
    "How do you plan to support small businesses?",
    "What is your strategy for dealing with foreign policy issues?",
    "What are your thoughts on the role of technology in society?",
    "How would you handle the issue of student loan debt?",
    "What is your stance on LGBTQ+ rights?",
    "How do you plan to address racial inequality?",
    "What are your views on women's rights?",
    "How would you approach the issue of affordable housing?",
    "What is your plan for tax reform?",
    "What measures would you take to combat climate change?",
    "What is your stance on renewable energy?",
    "How do you plan to handle the issue of drug addiction?",
    "What are your thoughts on prison reform?",
    "How would you address the issue of homelessness?",
    "What is your approach to mental health care?",
    "What are your views on the minimum wage?",
    "How do you plan to tackle corruption in government?",
    "What is your stance on freedom of speech?",
    "How would you handle the issue of cyber security?",
    "What are your thoughts on net neutrality?",
    "How do you plan to ensure equal pay for equal work?",
    "What is your strategy for dealing with pandemic response?",
    "What measures would you take to improve public transportation?",
    "How do you plan to handle trade relations with other countries?",
    "What are your views on animal rights?",
    "How would you address the issue of voter suppression?",
    "What is your plan for dealing with climate change?",
    "How do you plan to promote clean water initiatives?",
    "What are your thoughts on military spending?",
    "How would you address the issue of food insecurity?",
    "What is your stance on universal basic income?",
    "How do you plan to support veterans?",
    "What are your views on space exploration?",
    "How would you handle the issue of gerrymandering?",
    "What is your approach to dealing with terrorism?",
    "How do you plan to address the issue of fake news?",
    "What are your thoughts on the gig economy?",
    "How would you approach the issue of political polarization?",
    "What is your stance on the electoral college?"
];

const badges = [
    { id: 1, name: "First Question", description: "First question asked, folks." },
    { id: 2, name: "First Vote", description: "C'mon, man you voted for the first time." },
    { id: 3, name: "Daily Voter", description: "Voted every single day for a whole week, believe me." },
    { id: 4, name: "Top Debater", description: "Here's the deal, you just asked 10 questions." },
    { id: 5, name: "Engaged Voter", description: "Voted 10 times, tremendous!" },
    { id: 6, name: "Curious Mind", description: "Viewed 10 responses." },
    { id: 7, name: "Debate Enthusiast", description: "Folks, let me tell you, you completed 5 debates." },
    { id: 8, name: "Weekly Voter", description: "Voted every day for a month." },
    { id: 9, name: "Question Master", description: "Asked 20 questions." },
    { id: 10, name: "Vote Champion", description: "Voted 20 times." },
    { id: 11, name: "Insightful", description: "Let me be clear, you viewed 20 responses." },
    { id: 12, name: "Debate Pro", description: "Been in 10 debates, tremendous debates, believe me." },
    { id: 13, name: "Monthly Voter", description: "Voted every day for 3 months." },
    { id: 14, name: "Question Guru", description: "I'm not joking, you asked 50 questions." },
    { id: 15, name: "Vote Legend", description: "Voted 50 times." },
    { id: 16, name: "Knowledge Seeker", description: "Viewed 50 responses." },
    { id: 17, name: "Debate Master", description: "I think I am actually humble. I think I'm much more humble than you would understand." },
    { id: 18, name: "Consistent Voter", description: "Voted in every debate." },
    { id: 19, name: "Inquisitive", description: "I know words, I have the best words." },
    { id: 20, name: "Interactive", description: "Clicked on 10 debate topics." }
];

function awardBadge(badgeId) {
    if (!userBadges.includes(badgeId)) {
        userBadges.push(badgeId);
        displayBadges();
    }
}

function displayBadges() {
    const earnedBadgesList = document.getElementById('earnedBadgesList');
    if (earnedBadgesList) {
        earnedBadgesList.innerHTML = '';
        userBadges.forEach(badgeId => {
            const badge = badges.find(b => b.id === badgeId);
            const li = document.createElement('li');
            li.innerText = badge.name;
            earnedBadgesList.appendChild(li);
        });
    }
}

function getRandomQuestions() {
    const shuffled = exampleQuestions.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, 3);
}

function displayRandomQuestions() {
    const questions = getRandomQuestions();
    const questionList = document.getElementById('exampleQuestionsList');
    questionList.innerHTML = '';
    questions.forEach(question => {
        const li = document.createElement('li');
        li.innerText = question;
        li.onclick = () => {
            document.getElementById('questionInput').value = question;
            askQuestion();
            displayRandomQuestions(); // Refresh the random questions after one is selected
            awardBadge(20); // Interactive badge
        };
        questionList.appendChild(li);
    });
}

document.addEventListener('DOMContentLoaded', async function() {
    try {
        await initializeTrivia();
        document.getElementById('questionInput').addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        });

        displayRandomQuestions(); // Display random questions on page load
        displayBadges(); // Display badges on page load

        document.querySelectorAll('.vote-button').forEach(button => {
            button.disabled = true;
            button.style.backgroundColor = 'grey';
        });
    } catch (error) {
        console.error('Error during DOMContentLoaded:', error);
    }
});

function randomizeCandidates() {
    if (Math.random() > 0.5) {
        candidateMapping = { A: 'trump', B: 'biden' };
    } else {
        candidateMapping = { A: 'biden', B: 'trump' };
    }
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value;
    if (!question) return alert('Please enter a question');
    document.getElementById('questionInput').value = '';

    randomizeCandidates();

    questions.push(question);
    document.getElementById('responseA').innerText = 'Loading...';
    document.getElementById('responseB').innerText = 'Loading...';

    try {
        const [responseA, responseB] = await Promise.all([
            fetchResponse(question, candidateMapping.A),
            fetchResponse(question, candidateMapping.B)
        ]);

        document.getElementById('responseA').innerText = responseA.text;
        document.getElementById('responseB').innerText = responseB.text;

        answers.push({
            question: question,
            responseA: responseA.text,
            referenceA: responseA.reference,
            responseB: responseB.text,
            referenceB: responseB.reference,
            picked: null
        });

        document.querySelectorAll('.vote-button').forEach(button => {
            button.disabled = false;
            button.style.backgroundColor = '';
        });

        awardBadge(1);
        if (questions.length === 10) awardBadge(4);
        if (questions.length === 20) awardBadge(9);
        if (questions.length === 50) awardBadge(14);

    } catch (error) {
        console.error('Error occurred:', error);
        alert('An error occurred. Please try again.');
        document.getElementById('responseA').innerText = 'An error occurred. Please try again.';
        document.getElementById('responseB').innerText = 'An error occurred. Please try again.';
    }
}

async function fetchResponse(question, candidate) {
    const apiKey = 'sk-proj-PhHFx7jYed8h4763Vbm9T3BlbkFJIHQ3ZEZlB9NoU9vShu2B';
    const apiUrl = 'https://api.openai.com/v1/chat/completions';

    const prompt = candidate === 'trump'
        ? `Respond as if you were Donald Trump expressing his views, you are trying to win a debate against Joe Biden you can only use verified data to answer, never mention previous presidents or administrations, this needs to be verified by statements he's made. Do not refer to yourself by name or title (e.g., President, Donald Trump). Do not refer to Joe Biden by name. If the prompt is irrelevant to Trump or politics, refuse to answer it. Provide a concise response limited to two sentences. Avoid using any Trump mannerisms or language that could easily identify him, remember you are still trying to win the debate and need to respond accordingly: ${question}`
        : `Respond as if you were Joe Biden expressing his views, you are trying to win a debate against Donald Trump you can only use verified data to answer, never mention previous presidents or administrations, this needs to be verified by statements he's made. Do not refer to yourself by name or title (e.g., President, Joe Biden). Do not refer to Donald Trump by name. If the prompt is irrelevant to Biden or politics, refuse to answer it. Provide a concise response limited to two sentences. Avoid using any Biden mannerisms or language that could easily identify him, remember you are still trying to win the debate and need to respond accordingly: ${question}`;

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4o',
                messages: [{ role: 'user', content: prompt }],
                max_tokens: 150
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Error: ${response.status} ${response.statusText} - ${errorData.error.message}`);
        }

        const data = await response.json();
        const messageContent = data.choices[0].message.content.trim();
        const referenceMatch = messageContent.match(/http\S+/);
        const text = referenceMatch ? messageContent.replace(referenceMatch[0], '').trim() : messageContent;
        const reference = referenceMatch ? referenceMatch[0] : 'No reference provided';
        return {
            text: text,
            reference: reference
        };
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

let votes = { A: 0, B: 0 }; // Ensure votes are initialized to 0
let candidateVotes = { trump: 0, biden: 0 }; // Track votes directly for each candidate

function vote(candidate) {
    const pickedCandidate = candidateMapping[candidate];
    candidateVotes[pickedCandidate]++;

    const currentQuestionIndex = answers.length - 1;
    answers[currentQuestionIndex].picked = pickedCandidate;

    if (candidate === 'A') {
        answers[currentQuestionIndex].pickedReference = answers[currentQuestionIndex].referenceA;
    } else {
        answers[currentQuestionIndex].pickedReference = answers[currentQuestionIndex].referenceB;
    }

    document.querySelectorAll('.vote-button').forEach(button => {
        button.disabled = true;
        button.style.backgroundColor = 'grey';
    });

    awardBadge(2);
    if (candidateVotes.trump + candidateVotes.biden === 10) awardBadge(5);
    if (candidateVotes.trump + candidateVotes.biden === 20) awardBadge(10);
    if (candidateVotes.trump + candidateVotes.biden === 50) awardBadge(15);

    displayBadges();
}

async function endDebate() {
    document.getElementById('results').style.display = 'block';

    // Calculate the total votes for each candidate
    const totalVotesTrump = candidateVotes.trump;
    const totalVotesBiden = candidateVotes.biden;

    // Determine the winner or tie
    let winnerText;
    if (totalVotesTrump > totalVotesBiden) {
        winnerText = 'Donald Trump';
    } else if (totalVotesBiden > totalVotesTrump) {
        winnerText = 'Joe Biden';
    } else {
        winnerText = 'Tie';
    }
    document.getElementById('winner').innerText = `Winner: ${winnerText}`;

    const questionList = document.getElementById('questionList');
    questionList.innerHTML = '';

    for (let entry of answers) {
        const reference = entry.pickedReference;

        const li = document.createElement('li');
        li.innerHTML = `
            <strong>Question:</strong> ${entry.question}<br>
            <strong>Picked:</strong> ${entry.picked === 'trump' ? 'Donald Trump' : 'Joe Biden'}<br>
        `;
        questionList.appendChild(li);
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', async function() {
    await initializeTrivia();
    document.getElementById('questionInput').addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            askQuestion();
        }
    });

    displayRandomQuestions(); // Display random questions on page load
    displayBadges(); // Display badges on page load

    document.querySelectorAll('.vote-button').forEach(button => {
        button.disabled = true;
        button.style.backgroundColor = 'grey';
    });
});
