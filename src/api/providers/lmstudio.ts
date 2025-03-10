import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"

export class LmStudioHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			baseURL: (this.options.lmStudioBaseUrl || "http://localhost:1234") + "/v1",
			apiKey: "noop",
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		//console.log("LMStudio: Creating chat completion with model:", this.getModel().id);

		let inputTokens = 0
		let outputTokens = 0
		let chunkCount = 0
		let assistantMessageContent = ""

		try {
			const stream = await this.client.chat.completions.create({
				model: this.getModel().id,
				messages: openAiMessages,
				temperature: 0,
				stream: true,
				stream_options: { include_usage: true },
			})

			for await (const chunk of stream) {
				chunkCount++
				//console.log(`LMStudio: Received chunk #${chunkCount}:`, JSON.stringify(chunk));

				const delta = chunk.choices[0]?.delta

				// Capture token usage from the final chunk if available
				if (chunk.usage) {
					//console.log("LMStudio: Found usage in streaming chunk:", JSON.stringify(chunk.usage));
					inputTokens = chunk.usage.prompt_tokens || 0
					outputTokens = chunk.usage.completion_tokens || 0

					//console.log(`LMStudio: Yielding usage - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);
					yield {
						type: "usage",
						inputTokens,
						outputTokens,
						totalCost: 0, // LMStudio is free, so cost is 0
					}
				}

				if (delta?.content) {
					assistantMessageContent += delta.content
					yield {
						type: "text",
						text: delta.content,
					}
				}
				console.log(JSON.stringify(assistantMessageContent))
			}

			console.log(`================================`)
			//console.log(`LMStudio: Stream completed. Total chunks: ${chunkCount}. Token counts from stream: inputTokens=${inputTokens}, outputTokens=${outputTokens}`);

			// If we didn't get usage from the stream chunks, make a non-streaming request to get the usage
			// if (inputTokens === 0 && outputTokens === 0) {
			// 	//console.log("LMStudio: No token usage from stream, making non-streaming request to get usage");
			// 	try {
			// 		// First try the standard non-streaming request to get usage
			// 		const completion = await this.client.chat.completions.create({
			// 			model: this.getModel().id,
			// 			messages: openAiMessages,
			// 			temperature: 0,
			// 			stream: false,
			// 		});

			// 		console.log("LMStudio: Non-streaming completion response:", JSON.stringify(completion));

			// 		if (completion.usage) {
			// 			console.log("LMStudio: Found usage in non-streaming response:", JSON.stringify(completion.usage));
			// 			inputTokens = completion.usage.prompt_tokens || 0;
			// 			outputTokens = completion.usage.completion_tokens || 0;

			// 			console.log(`LMStudio: Yielding usage from non-streaming request - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);
			// 			yield {
			// 				type: "usage",
			// 				inputTokens,
			// 				outputTokens,
			// 				totalCost: 0, // LMStudio is free, so cost is 0
			// 			};
			// 		} else {
			// 			console.log("LMStudio: No usage information in non-streaming response, using estimation");

			// 			// If we still don't have token counts, use a rough estimation
			// 			// For LMStudio, we can estimate based on the text length
			// 			// This is a very rough estimate
			// 			const promptText = openAiMessages.map(msg => {
			// 				if (typeof msg.content === 'string') {
			// 					return msg.content;
			// 				} else if (Array.isArray(msg.content)) {
			// 					return msg.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
			// 				}
			// 				return '';
			// 			}).join('\n\n');

			// 			// Rough estimation: ~4 characters per token for English text
			// 			inputTokens = Math.ceil(promptText.length / 4);
			// 			outputTokens = Math.ceil(assistantMessageContent.length / 4);

			// 			console.log(`LMStudio: Estimated token counts - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);

			// 			yield {
			// 				type: "usage",
			// 				inputTokens,
			// 				outputTokens,
			// 				totalCost: 0, // LMStudio is free, so cost is 0
			// 			};
			// 		}
			// 	} catch (error) {
			// 		console.error("Failed to get token usage from LMStudio:", error);

			// 		// If the non-streaming request fails, fall back to estimation
			// 		console.log("LMStudio: Falling back to token estimation after error");

			// 		// Rough estimation: ~4 characters per token for English text
			// 		const promptText = openAiMessages.map(msg => {
			// 			if (typeof msg.content === 'string') {
			// 				return msg.content;
			// 			} else if (Array.isArray(msg.content)) {
			// 				return msg.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
			// 			}
			// 			return '';
			// 		}).join('\n\n');

			// 		inputTokens = Math.ceil(promptText.length / 4);
			// 		outputTokens = Math.ceil(assistantMessageContent.length / 4);

			// 		console.log(`LMStudio: Estimated token counts after error - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);

			// 		yield {
			// 			type: "usage",
			// 			inputTokens,
			// 			outputTokens,
			// 			totalCost: 0, // LMStudio is free, so cost is 0
			// 		};
			// 	}
			// }
		} catch (error) {
			console.error("LMStudio error:", error)
			// LM Studio doesn't return an error code/body for now
			throw new Error(
				"Please check the LM Studio developer logs to debug what went wrong. You may need to load the model with a larger context length to work with Cline's prompts.",
			)
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.lmStudioModelId || "",
			info: openAiModelInfoSaneDefaults,
		}
	}
}
