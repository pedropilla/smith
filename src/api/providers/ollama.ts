import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"

export class OllamaHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			baseURL: (this.options.ollamaBaseUrl || "http://localhost:11434") + "/v1",
			apiKey: "ollama",
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		//console.log(JSON.stringify(openAiMessages));
		//console.log("Ollama: Creating chat completion with model:", this.getModel().id);

		const stream = await this.client.chat.completions.create({
			model: this.getModel().id,
			messages: openAiMessages,
			temperature: 0,
			stream: true,
			stream_options: { include_usage: true },
		})

		let inputTokens = 0
		let outputTokens = 0
		let chunkCount = 0
		let assistantMessageContent = ""

		for await (const chunk of stream) {
			chunkCount++
			//console.log(`Ollama: Received chunk #${chunkCount}:`, JSON.stringify(chunk));

			const delta = chunk.choices[0]?.delta

			// Capture token usage from the final chunk if available
			if (chunk.usage) {
				//console.log("Ollama: Found usage in streaming chunk:", JSON.stringify(chunk.usage));
				inputTokens = chunk.usage.prompt_tokens || 0
				outputTokens = chunk.usage.completion_tokens || 0

				//console.log(`Ollama: Yielding usage - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);
				yield {
					type: "usage",
					inputTokens,
					outputTokens,
					totalCost: 0, // Ollama is free, so cost is 0
				}
			}

			if (delta?.content) {
				assistantMessageContent += delta.content
				yield {
					type: "text",
					text: delta.content,
				}
			}
		}

		console.log(JSON.stringify(assistantMessageContent))
		//console.log(`Ollama: Stream completed. Total chunks: ${chunkCount}. Token counts from stream: inputTokens=${inputTokens}, outputTokens=${outputTokens}`);

		// If we didn't get usage from the stream chunks, use a simple estimation approach
		// if (inputTokens === 0 && outputTokens === 0) {
		// 	console.log("Ollama: No token usage from stream, using character-based estimation");

		// 	try {
		// 		// Simple estimation based on character count
		// 		// This is a very rough estimate: ~4 characters per token for English text
		// 		const promptText = openAiMessages.map(msg => {
		// 			if (typeof msg.content === 'string') {
		// 				return msg.content;
		// 			} else if (Array.isArray(msg.content)) {
		// 				return msg.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
		// 			}
		// 			return '';
		// 		}).join('\n\n');

		// 		// Estimate tokens based on character count
		// 		inputTokens = Math.ceil(promptText.length / 4);
		// 		outputTokens = Math.ceil(assistantMessageContent?.length / 4) || 0;

		// 		console.log(`Ollama: Estimated token counts - inputTokens: ${inputTokens}, outputTokens: ${outputTokens}`);

		// 		// Try a non-streaming request as a fallback to get more accurate token counts
		// 		try {
		// 			console.log("Ollama: Attempting non-streaming request for token counts");
		// 			const completion = await this.client.chat.completions.create({
		// 				model: this.getModel().id,
		// 				messages: openAiMessages,
		// 				temperature: 0,
		// 				stream: false,
		// 			});

		// 			console.log("Ollama: Non-streaming completion response:", JSON.stringify(completion));

		// 			if (completion.usage) {
		// 				console.log("Ollama: Found usage in non-streaming response:", JSON.stringify(completion.usage));
		// 				inputTokens = completion.usage.prompt_tokens || inputTokens;
		// 				outputTokens = completion.usage.completion_tokens || outputTokens;
		// 			}
		// 		} catch (error) {
		// 			console.log("Ollama: Non-streaming request failed, using character-based estimates:", error);
		// 			// Continue with our character-based estimates
		// 		}

		// 		yield {
		// 			type: "usage",
		// 			inputTokens,
		// 			outputTokens,
		// 			totalCost: 0, // Ollama is free, so cost is 0
		// 		};
		// 	} catch (error) {
		// 		console.error("Failed to estimate token usage for Ollama:", error);

		// 		// Provide some minimal token counts to prevent context window issues
		// 		yield {
		// 			type: "usage",
		// 			inputTokens: 1000, // Conservative default estimate
		// 			outputTokens: 500,  // Conservative default estimate
		// 			totalCost: 0,       // Ollama is free, so cost is 0
		// 		};
		// 	}
		// }
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.ollamaModelId || "",
			info: openAiModelInfoSaneDefaults,
		}
	}
}
