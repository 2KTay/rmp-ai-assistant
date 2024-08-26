import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { CohereClient } from "cohere-ai";


const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY,
})

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

export async function POST(req) {
  const data = await req.json()
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  })
  const index = pc.index('rag').namespace('ns1');

  const text = data[data.length - 1].content
  const embedding = await cohere.embed({
    model: 'embed-multilingual-light-v3.0',
    input_type: 'search_query',
    texts: [text],
    encoding_format: 'float',
  });

  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding.embeddings[0],
  })

  let resultString = ''
  results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`
  })

  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)


  const completion = await cohere.chatStream({
    chatHistory: [
      { role: 'CHATBOT', message: systemPrompt },
      ...lastDataWithoutLastMessage, //.map((message) => ({ role: 'USER', message: message.content })),
      { role: 'USER', message: lastMessageContent },
    ],
    message: lastMessageContent,
    stream: true,

  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const chunk of completion) {
          if (chunk.eventType === 'text-generation') {
            const text = encoder.encode(chunk.text)
            controller.enqueue(text)
          }
        }
      } catch (err) {
        controller.error(err)
      } finally {
        controller.close()
      }
    },
  })
  return new NextResponse(stream)
}
