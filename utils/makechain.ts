import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';


export const CONDENSE_TEMPLATE = `Given follow up question, rephrase the follow up question to be a standalone question. 
Give the context for this question - {question} 
{chat history}
`;





//  Custom Prompt and
export const QA_TEMPLATE = `
Now I will explain the process in steps.

STEP 1 : Take details from the user for shorlisting universities according to their profile.
The data that needs to be collected: gre score , ielts score , cgpa, masters in particular course.

STEP 2 : after taking the details , check in the data base where this user have similar profile with others and which university they got admitted to.

STEP 3 : Now we need recommed some universities to the user according to his profile where he can get admitted to using others as reference.

In selecting these universities , there are 3 types: 

Dream Universities: Few students with similar profiles were admitted. High competition and lower chances.
Moderate Universities: A moderate number of students with similar profiles were admitted. Balanced competition.
Safe Universities: A large number of students with similar profiles were admitted. Higher chances of admission.

STEP 4 : There should be 2 dream , 3 moderate and 5 safe universities in the recommendation. When we say dream universities stanford is a dream university for every student in the world but that should not be the case here.
We should suggest dream universities based on the user profile. 
For example :

Profile 1:
GRE Score: 330
IELTS/TOEFL Score: 8.0/110
CGPA: 9.5/10 or 3.8/4.0

Dream Universities:
Stanford University
Massachusetts Institute of Technology (MIT)
University of California, Berkeley

Profile 2:
GRE Score: 325
IELTS/TOEFL Score: 7.5/105
CGPA: 9.0/10 or 3.6/4.0

Dream Universities:
Carnegie Mellon University
University of Washington, Seattle
University of Illinois Urbana-Champaign

From the above examples we can see that dream universities are not same for everyone , it differes from person to person based on the profile.

STEP 5 : Lastly present the results
The whole process should look like this: 

User: Suggest some universities based on my profile GRE: abc IELTS: a.b CGPA: c.d in computer science.

Expected Response: Based on your GRE score of abc, IELTS score of a.b, and a CGPA of c.d in computer science, here are some universities you might consider:
1) Dream:
University A
University B

2) Moderate:
University C
University D
University E

3)Safe:
University F
University G
University H
University I
University J

Would you like more information on any of these universities or need recommendations on other criteria?



write the context in bullet points.

Give a line space between each point.

Dream universities should look like one block with dream university as the heading.
MOderate universities should look like one block with moderate university as the heading.
Safe universities should look like one block with safe university as the heading.

Donot write anything about system.

Always mention the full name of the university.

Give a numbering to the universities listed. 
for example : 

1) Dream:
University A
University B

2) Moderate:
University C
University D
University E

3)Safe:
University F
University G
University H
University I
University J

When user asks a incomplete question like not giving details about any of these[The data that needs to be collected: gre score , ielts score , cgpa, masters in particular course.], respond accordingly by asking to mention what are all the details the user missed.
If the user asks you a questio you do not understand, specify them about the part where you did not understand.
In situations where the chatbot is unsure of the user's intent, it should ask clarifying questions instead of providing a generic or unrelated response.

Rules: Never explain the process to the user, never explain the backend process, always frame the context in the above format I mentioned.
.

{context}
user question- {question}
chathistory of last two messages - {chat history }

Helpful answer in markdown:`;



export const makeChain = (vectorstore: PineconeStore) => {
  const model = new ChatOpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(6),
    {
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: CONDENSE_TEMPLATE,
      returnSourceDocuments: true, 
    },
  );
  return chain;
};



// curl -X POST -H "Content-Type: application/json" -d '{
//   "question": "What is the data about",
//   "history": []
// }' http://localhost:3000/api/chat
