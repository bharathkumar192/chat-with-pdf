import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
export var CONDENSE_TEMPLATE_ = '';
export var QA_TEMPLATE_ = '';


async function fetchCustomPrompt() {
  const response = await fetch('https://raw.githubusercontent.com/bharathkumar192/azure-sample-test/main/prompt.json');
  const data: { CONDENSE_TEMPLATE: { [key: string]: string[] }, QA_TEMPLATE: { [key: string]: string[] } } = await response.json();
  const condenseTemplateString = Object.values(data.CONDENSE_TEMPLATE).map((section: string[]) => section.join('\n')).join('\n');
  const qaTemplateString = Object.values(data.QA_TEMPLATE).map((section: string[]) => section.join('\n')).join('\n');
  return { 
      condenseTemplate: condenseTemplateString,
      qaTemplate: qaTemplateString
  };
}
fetchCustomPrompt().then(result => {
  CONDENSE_TEMPLATE_ = result.condenseTemplate;
  QA_TEMPLATE_ = result.qaTemplate;
});


export const makeChain = (vectorstore: PineconeStore) => {
  const model = new ChatOpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(6),
    {
      qaTemplate: QA_TEMPLATE_,
      questionGeneratorTemplate: CONDENSE_TEMPLATE_,
      returnSourceDocuments: true, 
    },
  );
  return chain;
};



// curl -X POST -H "Content-Type: application/json" -d '{
//   "question": "What is the data about",
//   "history": []
// }' http://localhost:3000/api/chat
