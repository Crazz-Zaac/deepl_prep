// generate-questions.js
// This script generates the questions.json file from the questions database

const fs = require('fs');
const questionsDatabase = require('./questions-database.js');

// Function to generate the questions.json file
function generateQuestionsJson() {
  let id = 1;
  const allQuestions = [];

  // Process single choice questions
  questionsDatabase.singleChoice.forEach(q => {
    allQuestions.push({
      id: id++,
      type: "single",
      question: q.question,
      options: q.options,
      answer: q.answer
    });
  });

  // Process multiple choice questions
  questionsDatabase.multipleChoice.forEach(q => {
    allQuestions.push({
      id: id++,
      type: "multiple",
      question: q.question,
      options: q.options,
      answer: q.answer
    });
  });

  // Process short answer questions
  questionsDatabase.shortAnswer.forEach(q => {
    allQuestions.push({
      id: id++,
      type: "short",
      question: q.question,
      answer: q.answer
    });
  });

  // Write to questions.json
  fs.writeFileSync('questions.json', JSON.stringify(allQuestions, null, 2));
  console.log(`Successfully generated questions.json with ${allQuestions.length} questions`);
}

// Run the generator
generateQuestionsJson();