import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export interface FashionAnalysis {
  color: string;
  pattern: string;
  style: string;
  category: string;
  description: string;
  hairStyle?: string;
  hairColor?: string;
}

export interface Recommendation {
  name: string;
  category: string;
  reason: string;
  platform: string;
  priceRange: string;
  imageUrl: string;
  purchaseUrl: string;
  matchScore: number;
}

export const analyzeFashionItem = async (base64Image: string): Promise<{ analysis: FashionAnalysis, recommendations: Recommendation[] }> => {
  const model = "gemini-3-flash-preview"; 
  
  const prompt = `Analyze this clothing item and provide:
  1. Detailed attributes (color, pattern, style, category).
  2. If a person is visible, analyze their hair (style, color, length).
  3. 3-4 matching outfit recommendations (e.g., if it's a shirt, suggest pants, shoes, and an accessory).
  
  CRITICAL: For each recommendation, you MUST provide a REAL, VALID purchase link. 
  Use Google Search to find current products on Myntra, Amazon.in, Ajio, or Flipkart.
  
  If you cannot find a direct product page link, CONSTRUCT a valid search URL using these templates:
  - Ajio: https://www.ajio.com/search/?text=[ITEM_NAME_AND_COLOR]
  - Myntra: https://www.myntra.com/[ITEM_NAME_AND_COLOR_HYPHENATED]
  - Amazon: https://www.amazon.in/s?k=[ITEM_NAME_AND_COLOR]
  - Flipkart: https://www.flipkart.com/search?q=[ITEM_NAME_AND_COLOR]
  
  Replace [ITEM_NAME_AND_COLOR] with the suggested item's name and color.
  
  Provide a realistic item name, the platform, a price range in INR, and a match score (0-100).
  Also provide a reason why it matches.`;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: base64Image.split(",")[1] || base64Image,
              },
            },
          ],
        },
      ],
      config: {
        tools: [{ googleSearch: {} }],
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            analysis: {
              type: Type.OBJECT,
              properties: {
                color: { type: Type.STRING },
                pattern: { type: Type.STRING },
                style: { type: Type.STRING },
                category: { type: Type.STRING },
                description: { type: Type.STRING },
                hairStyle: { type: Type.STRING, description: "The user's hairstyle if visible" },
                hairColor: { type: Type.STRING, description: "The user's hair color if visible" },
              },
              required: ["color", "pattern", "style", "category", "description"],
            },
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  category: { type: Type.STRING },
                  reason: { type: Type.STRING },
                  platform: { type: Type.STRING },
                  priceRange: { type: Type.STRING },
                  imageUrl: { type: Type.STRING },
                  purchaseUrl: { type: Type.STRING },
                  matchScore: { type: Type.NUMBER },
                },
                required: ["name", "category", "reason", "platform", "priceRange", "imageUrl", "purchaseUrl", "matchScore"],
              },
            },
          },
          required: ["analysis", "recommendations"],
        },
      },
    });

    const data = JSON.parse(response.text || "{}");
    
    // Post-process recommendations to ensure valid URLs and generate matching images in parallel
    if (data.recommendations && Array.isArray(data.recommendations)) {
      data.recommendations = await Promise.all(data.recommendations.map(async (rec: Recommendation) => {
        let url = rec.purchaseUrl;
        const query = encodeURIComponent(`${rec.name} ${rec.category}`);
        
        // 1. Fix URLs
        if (!url || !url.startsWith('http') || url.includes('example.com')) {
          const platform = rec.platform.toLowerCase();
          if (platform.includes('ajio')) {
            url = `https://www.ajio.com/search/?text=${query}`;
          } else if (platform.includes('myntra')) {
            url = `https://www.myntra.com/search?q=${query}`;
          } else if (platform.includes('amazon')) {
            url = `https://www.amazon.in/s?k=${query}`;
          } else if (platform.includes('flipkart')) {
            url = `https://www.flipkart.com/search?q=${query}`;
          } else {
            url = `https://www.google.com/search?q=${query}+buy+online`;
          }
        }

        // 2. Generate matching image (we keep this async but it's now part of Promise.all)
        let generatedImageUrl = rec.imageUrl;
        try {
          const imageGenResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
              parts: [
                {
                  text: `A high-end, professional fashion product photograph of ${rec.name} (${rec.category}). Minimalist luxury studio background, soft lighting, high fashion aesthetic, 8k resolution.`,
                },
              ],
            },
            config: {
              imageConfig: {
                aspectRatio: "1:1",
              },
            },
          });

          for (const part of imageGenResponse.candidates[0].content.parts) {
            if (part.inlineData) {
              generatedImageUrl = `data:image/png;base64,${part.inlineData.data}`;
              break;
            }
          }
        } catch (err: any) {
          const isQuotaError = err.message?.includes("429") || err.status === "RESOURCE_EXHAUSTED" || (err.error && err.error.code === 429);
          if (!isQuotaError) {
            console.error("Image generation failed for recommendation:", err);
          }
          // Always use fallback for recommendations if generation fails for any reason
          generatedImageUrl = `https://picsum.photos/seed/${encodeURIComponent(rec.name)}/400/400`;
        }

        return { ...rec, purchaseUrl: url, imageUrl: generatedImageUrl };
      }));
    }

    return data;
  } catch (err: any) {
    console.error("Fashion analysis failed:", err);
    if (err.message?.includes("429") || err.status === "RESOURCE_EXHAUSTED") {
      throw new Error("STYLING_QUOTA_EXCEEDED");
    }
    throw err;
  }
};

export const getStylistResponse = async (message: string, history: { role: string, content: string, userImageUrl?: string }[], base64Image?: string) => {
  const model = "gemini-3-flash-preview";
  
  // Map history to the format expected by generateContent
  // We strip images from history to keep the context window clean and avoid redundancy,
  // as we will include the most relevant image in the current turn.
  const contents: any[] = history.map((h) => {
    return { 
      role: h.role === 'user' ? 'user' : 'model', 
      parts: [{ text: h.content }] 
    };
  });

  const userParts: any[] = [{ text: message }];
  
  // Find the most relevant image: either the one just uploaded, or the most recent one in history
  let imageToInclude = base64Image;
  if (!imageToInclude) {
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].userImageUrl) {
        imageToInclude = history[i].userImageUrl;
        break;
      }
    }
  }

  if (imageToInclude) {
    userParts.push({
      inlineData: {
        mimeType: "image/jpeg",
        data: imageToInclude.split(",")[1] || imageToInclude,
      }
    });
  }
  
  contents.push({ role: 'user', parts: userParts });

  try {
    const response = await ai.models.generateContent({
      model,
      contents,
      config: {
        systemInstruction: `You are LUMIÈRE, a high-end luxury fashion stylist. 
        Your tone is warm, sophisticated, and encouraging. 
        
        CRITICAL: You MUST analyze any image provided by the user. 
        Identify the clothing items, colors, patterns, textures, and the user's physical features (like hair and face shape) to provide highly personalized styling advice.
        If an image is present, your response should be directly based on what you see in that image.
        
        Always provide a comprehensive styling breakdown for every request:
        1. Provide a 'friendlyResponse' using Markdown. Use bold text for key items and bullet points for clarity. 
           Structure it strictly like: 
           - **The Vision**: (overall vibe based on the image)
           - **The Components**: (Top, Bottom, Shoes, Accessories)
           - **The Hairstyle**: Analyze the user's hair in the provided image (length, texture, color) and suggest a hairstyle that complements their face shape and the outfit. If no image is provided, suggest a versatile look.
           - **Stylist Tip**: (pro advice)
        2. Provide a 'visualPrompt': A VERY DETAILED, descriptive prompt for an image generation model of the COMPLETE OUTFIT. Include lighting, fabric textures, and setting.
        3. Provide a 'hairVisualPrompt': A VERY DETAILED, descriptive prompt for an image generation model focusing ONLY on the HAIRSTYLE (close-up). It MUST be inspired by or a refined version of the user's actual hair seen in the photo (if provided). Include hair texture, color, and lighting.
        4. Provide 'recommendations': A list of 2-3 specific items that would complete this look, including real-world platforms (Ajio, Myntra, Amazon.in).`,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            friendlyResponse: { type: Type.STRING },
            visualPrompt: { type: Type.STRING },
            hairVisualPrompt: { type: Type.STRING },
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  category: { type: Type.STRING },
                  reason: { type: Type.STRING },
                  platform: { type: Type.STRING },
                  priceRange: { type: Type.STRING },
                  purchaseUrl: { type: Type.STRING },
                },
                required: ["name", "category", "reason", "platform", "priceRange", "purchaseUrl"]
              }
            }
          },
          required: ["friendlyResponse", "visualPrompt", "hairVisualPrompt", "recommendations"]
        }
      }
    });

    const data = JSON.parse(response.text || "{}");
    const friendlyText = data.friendlyResponse;
    const visualPrompt = data.visualPrompt;
    const hairVisualPrompt = data.hairVisualPrompt;
    const rawRecommendations = data.recommendations || [];

    let imageUrl = "";
    let hairImageUrl = "";

    // Parallelize image generation and recommendation processing
    const imagePromises = [];

    if (visualPrompt && visualPrompt.trim().length > 0) {
      imagePromises.push((async () => {
        try {
          const outfitImageResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
              parts: [{ text: `A high-end, professional fashion editorial photograph of a complete outfit: ${visualPrompt}. Minimalist luxury studio, 8k, highly detailed.` }],
            },
            config: { imageConfig: { aspectRatio: "3:4" } },
          });
          for (const part of outfitImageResponse.candidates[0].content.parts) {
            if (part.inlineData) return { type: 'outfit', data: `data:image/png;base64,${part.inlineData.data}` };
          }
        } catch (e: any) { 
          const isQuotaError = e.message?.includes("429") || e.status === "RESOURCE_EXHAUSTED" || (e.error && e.error.code === 429);
          if (isQuotaError) return { type: 'outfit', data: null, error: 'QUOTA_EXCEEDED' };
          console.error("Outfit image generation failed", e);
        }
        return null;
      })());
    }

    if (hairVisualPrompt && hairVisualPrompt.trim().length > 0) {
      imagePromises.push((async () => {
        try {
          const hairImageResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
              parts: [{ text: `A high-end, professional beauty photography close-up of a hairstyle: ${hairVisualPrompt}. Minimalist luxury studio, soft lighting, 8k, highly detailed.` }],
            },
            config: { imageConfig: { aspectRatio: "1:1" } },
          });
          for (const part of hairImageResponse.candidates[0].content.parts) {
            if (part.inlineData) return { type: 'hair', data: `data:image/png;base64,${part.inlineData.data}` };
          }
        } catch (e: any) { 
          const isQuotaError = e.message?.includes("429") || e.status === "RESOURCE_EXHAUSTED" || (e.error && e.error.code === 429);
          if (isQuotaError) return { type: 'hair', data: null, error: 'QUOTA_EXCEEDED' };
          console.error("Hair image generation failed", e);
        }
        return null;
      })());
    }

    // Process recommendations in parallel with images
    const processedRecommendationsPromise = Promise.all(rawRecommendations.map(async (rec: any) => {
      const query = encodeURIComponent(`${rec.name} ${rec.category}`);
      let url = rec.purchaseUrl;
      
      if (!url || !url.startsWith('http') || url.includes('example.com')) {
        const platform = rec.platform.toLowerCase();
        if (platform.includes('ajio')) url = `https://www.ajio.com/search/?text=${query}`;
        else if (platform.includes('myntra')) url = `https://www.myntra.com/search?q=${query}`;
        else if (platform.includes('amazon')) url = `https://www.amazon.in/s?k=${query}`;
        else if (platform.includes('flipkart')) url = `https://www.flipkart.com/search?q=${query}`;
        else url = `https://www.google.com/search?q=${query}+buy+online`;
      }

      let recImageUrl = `https://picsum.photos/seed/${encodeURIComponent(rec.name)}/400/400`;
      try {
        const recImgResponse = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: { parts: [{ text: `A high-end fashion product shot of ${rec.name} ${rec.category}. Luxury studio, 8k.` }] },
          config: { imageConfig: { aspectRatio: "1:1" } }
        });
        for (const part of recImgResponse.candidates[0].content.parts) {
          if (part.inlineData) {
            recImageUrl = `data:image/png;base64,${part.inlineData.data}`;
            break;
          }
        }
      } catch (e) {}

      return { ...rec, purchaseUrl: url, imageUrl: recImageUrl };
    }));

    const [imageResults, processedRecommendations] = await Promise.all([
      Promise.all(imagePromises),
      processedRecommendationsPromise
    ]);

    let quotaExceeded = false;
    imageResults.forEach(res => {
      if (!res) return;
      if (res.error === 'QUOTA_EXCEEDED') quotaExceeded = true;
      if (res.type === 'outfit') imageUrl = res.data;
      if (res.type === 'hair') hairImageUrl = res.data;
    });

    let finalResponse = friendlyText;
    if (quotaExceeded) {
      finalResponse += "\n\n*(Note: I've reached my visual generation limit for the moment, so I've provided high-quality style references instead. I'll be back to full creative power shortly!)*";
      if (!imageUrl) imageUrl = `https://picsum.photos/seed/${encodeURIComponent(visualPrompt || 'fashion')}/800/1200`;
      if (!hairImageUrl) hairImageUrl = `https://picsum.photos/seed/${encodeURIComponent(hairVisualPrompt || 'hair')}/800/800`;
    }

    return { text: finalResponse, imageUrl, hairImageUrl, recommendations: processedRecommendations };
  } catch (err: any) {
    const isQuotaError = err.message?.includes("429") || err.status === "RESOURCE_EXHAUSTED" || (err.error && err.error.code === 429);
    if (!isQuotaError) {
      console.error("Stylist response failed:", err);
    }
    
    if (isQuotaError) {
      return { 
        text: "I'm sorry, but I've reached my daily styling limit. Please try again in a little while, and I'll be happy to help you with your fashion needs!", 
        imageUrl: "", 
        hairImageUrl: "" 
      };
    }
    throw err;
  }
};
