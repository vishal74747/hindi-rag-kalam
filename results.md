# Test query results

embedding model: `BAAI/bge-m3` · retrieval: `hybrid (keyword weight 1.0) + query expansion` · llm: `groq` · top-k: 4

```text
Query : SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?
Answer: SLV-III ने रोहिणी उपग्रह को कक्षा में स्थापित किया, यह कार्य जुलाई 1980 में पूरा हुआ था[28][60].
Sources:
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 28 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 27 · rrf: 0.032 (dense #2, bm25 #3)
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 25 · rrf: 0.032 (dense #3, bm25 #2)
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 60 · rrf: 0.0312 (dense #4, bm25 #4)  <- cited in answer

Query : कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?
Answer: कलाम को भारत रत्न 1997 में प्राप्त हुआ [58][69]
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 57 · rrf: 0.0325 (dense #2, bm25 #1)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.032 (dense #1, bm25 #4)  <- cited in answer
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0313 (dense #5, bm25 #3)  <- cited in answer
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0308 (dense #4, bm25 #6)

Query : पोखरण-II में कलाम की भूमिका क्या थी?
Answer: पोखरण-II में कलाम उस समय रक्षा मंत्री के वैज्ञानिक सलाहकार और डीआरडीओ के प्रमुख थे; उन्होंने परमाणु ऊर्जा विभाग के वैज्ञानिकों के साथ मिलकर एक केंद्रीय संगठनात्मक भूमिका निभाई[34].
Sources:
  - page: 10 · section: "पोखरण-II: 1998 के परमाणु परीक्षण" · chunk_id: 34 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0318 (dense #2, bm25 #4)
  - page: 10 · section: "पोखरण-II: 1998 के परमाणु परीक्षण" · chunk_id: 35 · rrf: 0.0317 (dense #3, bm25 #3)
  - page: 10 · section: "पोखरण-II: 1998 के परमाणु परीक्षण" · chunk_id: 36 · rrf: 0.031 (dense #4, bm25 #5)

Query : Which institution did Kalam attend to study aeronautical engineering?
Answer: He studied aeronautical engineering at the Madras Institute of Technology (MIT) in Chennai[17][18].
Sources:
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0313 (dense #1, bm25 #7)  <- cited in answer
  - page: 1 · section: "शीर्षक पृष्ठ" · chunk_id: 0 · rrf: 0.0301 (dense #13, bm25 #1)
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 18 · rrf: 0.0294 (dense #3, bm25 #14)  <- cited in answer
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 59 · rrf: 0.0291 (dense #17, bm25 #2)

Query : Who co-wrote the autobiography, and in what year was it published?
Answer: The autobiography “Wings of Fire” was co-written with scientist Arun Tiwari and was published in 1999[5].
Sources:
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 5 · rrf: 0.0325 (dense #2, bm25 #1)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 6 · rrf: 0.0323 (dense #1, bm25 #3)
  - page: 1 · section: "शीर्षक पृष्ठ" · chunk_id: 0 · rrf: 0.0315 (dense #3, bm25 #4)
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 60 · rrf: 0.0313 (dense #6, bm25 #2)

Query : How and where did Kalam die in 2015?
Answer: Dr. A. P. J. Abdul Kalam died on 27 July 2015 after collapsing while giving a lecture in Shillong, Meghalaya, India at the age of 83[55][74].
Sources:
  - page: 15 · section: "अंतिम क्षण तक शिक्षक (2007–2015)" · chunk_id: 55 · rrf: 0.0325 (dense #1, bm25 #2)  <- cited in answer
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 61 · rrf: 0.0318 (dense #5, bm25 #1)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.0315 (dense #2, bm25 #5)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.031 (dense #3, bm25 #6)

Query : What was Kalam's favourite food?   [not in the document: a grounded answer says so]
Answer: The provided passages do not mention Kalam’s favourite food.
Sources:
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0311 (dense #1, bm25 #8)
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0308 (dense #5, bm25 #5)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0306 (dense #8, bm25 #3)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 10 · rrf: 0.03 (dense #2, bm25 #12)

Query : कलाम की पत्नी का नाम क्या था?   [not in the document: a grounded answer says so]
Answer: इन दिए गए अंशों में कलाम की पत्नी के नाम का उल्लेख नहीं है।
Sources:
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0309 (dense #1, bm25 #9)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.0308 (dense #7, bm25 #3)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0295 (dense #4, bm25 #12)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 8 · rrf: 0.0289 (dense #2, bm25 #18)

Query : कलाम को पद्म भूषण किस वर्ष मिला?
Answer: कलाम को पद्म भूषण 1981 में मिला। [58]
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0323 (dense #3, bm25 #1)  <- cited in answer
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0304 (dense #10, bm25 #2)
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0303 (dense #7, bm25 #5)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0291 (dense #2, bm25 #17)

Query : In which year did Kalam receive the Padma Bhushan?
Answer: Kalam received the Padma Bhushan in 1981[58].
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 57 · rrf: 0.032 (dense #3, bm25 #2)
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0315 (dense #6, bm25 #1)  <- cited in answer
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0303 (dense #7, bm25 #5)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0292 (dense #1, bm25 #18)

Query : कलाम को भारत रत्न 2005 में मिला था, है ना?
Answer: नहीं, कलाम को भारत रत्न 1997 में मिला था [69][58]
Sources:
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0315 (dense #4, bm25 #3)  <- cited in answer
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0304 (dense #10, bm25 #2)
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0302 (dense #3, bm25 #10)

Query : Kalam died in Chennai, right?
Answer: No, Kalam died on 27 July 2015 in Shillong, Meghalaya, not in Chennai. [74]
Sources:
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0315 (dense #5, bm25 #2)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.0313 (dense #3, bm25 #5)  <- cited in answer
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0303 (dense #8, bm25 #4)
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0297 (dense #4, bm25 #11)

Query : In which years did Kalam receive the Padma Bhushan and the Bharat Ratna?
Answer: Kalam was awarded the Padma Bhushan in 1981 and the Bharat Ratna in 1997[58].
Sources:
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0323 (dense #1, bm25 #3)
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 57 · rrf: 0.0323 (dense #2, bm25 #2)
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0318 (dense #5, bm25 #1)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0296 (dense #3, bm25 #13)

Query : रोहिणी उपग्रह किस वर्ष स्थापित हुआ और कलाम का निधन किस शहर में हुआ?
Answer: रोहिणी उपग्रह 1980 में स्थापित हुआ और कलाम का निधन शिलांग शहर में हुआ। [28][74]
Sources:
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 28 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 27 · rrf: 0.0323 (dense #2, bm25 #2)
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 60 · rrf: 0.0313 (dense #5, bm25 #3)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.03 (dense #3, bm25 #11)  <- cited in answer

Query : कलाम के कितने बच्चे थे?   [not in the document: a grounded answer says so]
Answer: उपलब्ध संदर्भों में कलाम के बच्चों की संख्या का उल्लेख नहीं है।
Sources:
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.031 (dense #4, bm25 #5)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0306 (dense #3, bm25 #8)
  - page: 12 · section: "गुरु एवं शिक्षक" · chunk_id: 44 · rrf: 0.0305 (dense #11, bm25 #1)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 10 · rrf: 0.03 (dense #2, bm25 #12)

Query : What is Kalam's favourite colour?   [not in the document: a grounded answer says so]
Answer: The provided passages do not contain any information about Kalam’s favourite colour.
Sources:
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0313 (dense #5, bm25 #3)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0306 (dense #3, bm25 #8)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 10 · rrf: 0.03 (dense #2, bm25 #12)
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0297 (dense #10, bm25 #5)

Query : कलाम का जन्म कहाँ और कब हुआ?
Answer: अवुल पाकिर जैनुलाब्दीन अब्दुल कलाम का जन्म 15 अक्टूबर 1931 को तमिलनाडु के रामेश्वरम में हुआ था[8][74].
Sources:
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 8 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.0323 (dense #2, bm25 #2)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0303 (dense #4, bm25 #8)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0295 (dense #5, bm25 #11)

Query : Which college did Kalam study physics at?
Answer: He studied physics at St. Joseph’s College in Tiruchirappalli [17][74].
Sources:
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0315 (dense #3, bm25 #4)
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 18 · rrf: 0.0313 (dense #2, bm25 #6)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 74 · rrf: 0.0313 (dense #5, bm25 #3)  <- cited in answer

Query : सतीश धवन ने 1979 की प्रक्षेपण विफलता पर क्या किया?
Answer: सतीश धवन ने सार्वजनिक रूप से उस 1979 के प्रक्षेपण की विफलता की ज़िम्मेदारी स्वयं ली और अपनी टीम की रक्षा की[26][27].
Sources:
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 26 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 12 · section: "गुरु एवं शिक्षक" · chunk_id: 45 · rrf: 0.032 (dense #3, bm25 #2)
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 27 · rrf: 0.0315 (dense #2, bm25 #5)  <- cited in answer
  - page: 11 · section: "अग्नि की उड़ान के मुख्य विषय" · chunk_id: 40 · rrf: 0.0313 (dense #5, bm25 #3)

Query : What type of missile is Prithvi?
Answer: Prithvi is a short-range surface-to-surface missile[32].
Sources:
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 32 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 33 · rrf: 0.0323 (dense #2, bm25 #2)
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 30 · rrf: 0.0317 (dense #3, bm25 #3)
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 60 · rrf: 0.0305 (dense #5, bm25 #6)

Query : नाग मिसाइल किस प्रकार की है?
Answer: नाग मिसाइल एक टैंक-रोधी निर्देशित मिसाइल है। [33]
Sources:
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 33 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 32 · rrf: 0.0323 (dense #2, bm25 #2)
  - page: 9 · section: "निर्देशित मिसाइल कार्यक्रम" · chunk_id: 30 · rrf: 0.0317 (dense #3, bm25 #3)
  - page: 22 · section: "त्वरित तथ्य-सारांश" · chunk_id: 75 · rrf: 0.0297 (dense #5, bm25 #10)

Query : कलाम को पद्म विभूषण किस वर्ष मिला?
Answer: कलाम को पद्म विभूषण 1990 में मिला। [58]
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0323 (dense #3, bm25 #1)  <- cited in answer
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0305 (dense #6, bm25 #5)
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.0304 (dense #10, bm25 #2)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0291 (dense #2, bm25 #17)

Query : Who headed ISRO when the 1979 launch failed, and what did he do?
Answer: Satish Dhawan was the head of ISRO at the time of the 1979 launch failure, and he publicly took responsibility for the failure and defended his team [26][45].
Sources:
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 26 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 17 · section: "जीवन-रेखा (टाइमलाइन)" · chunk_id: 59 · rrf: 0.0312 (dense #4, bm25 #4)
  - page: 11 · section: "अग्नि की उड़ान के मुख्य विषय" · chunk_id: 40 · rrf: 0.0306 (dense #9, bm25 #2)
  - page: 12 · section: "गुरु एवं शिक्षक" · chunk_id: 45 · rrf: 0.0305 (dense #6, bm25 #5)  <- cited in answer

Query : कलाम किस वर्ष भारत के राष्ट्रपति चुने गए?
Answer: कलाम 2002 में भारत के राष्ट्रपति चुने गए थे। [49]
Sources:
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 51 · rrf: 0.0315 (dense #3, bm25 #4)
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0311 (dense #2, bm25 #7)
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 52 · rrf: 0.0297 (dense #4, bm25 #11)

Query : What did young Kalam do during World War II to earn money?
Answer: He collected bundles of newspapers that were thrown from moving trains and distributed them in the town, earning his own money during World War II[11].
Sources:
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 11 · rrf: 0.0328 (dense #1, bm25 #1)  <- cited in answer
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0305 (dense #6, bm25 #5)
  - page: 8 · section: "इसरो के वर्ष और SLV-III" · chunk_id: 25 · rrf: 0.0268 (dense #20, bm25 #10)
  - page: 4 · section: "रामेश्वरम में बचपन" · chunk_id: 9 · rrf: 0.0161 (dense #2, bm25 -)

Query : Kalam won the Bharat Ratna in 1981, correct?
Answer: No; A.P.J. Abdul Kalam was awarded the Bharat Ratna in 1997, not in 1981[69][58].
Sources:
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0325 (dense #1, bm25 #2)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0315 (dense #2, bm25 #5)
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0311 (dense #8, bm25 #1)  <- cited in answer
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 4 · rrf: 0.0301 (dense #5, bm25 #8)

Query : कलाम को नोबेल पुरस्कार किस वर्ष मिला?   [not in the document: a grounded answer says so]
Answer: दिए गए अंशों में कलाम को नोबेल पुरस्कार मिलने का कोई उल्लेख नहीं है।
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 57 · rrf: 0.0325 (dense #1, bm25 #2)
  - page: 6 · section: "कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 17 · rrf: 0.031 (dense #6, bm25 #3)
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0299 (dense #14, bm25 #1)
  - page: 20 · section: "संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 69 · rrf: 0.0289 (dense #2, bm25 #18)

Query : What was Kalam's salary as president?   [not in the document: a grounded answer says so]
Answer: The provided passages do not contain information about Kalam’s salary as president.
Sources:
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 49 · rrf: 0.0325 (dense #2, bm25 #1)
  - page: 3 · section: "ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · rrf: 0.0318 (dense #1, bm25 #5)
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 52 · rrf: 0.0312 (dense #4, bm25 #4)
  - page: 14 · section: "जनता के राष्ट्रपति (2002–2007)" · chunk_id: 51 · rrf: 0.0303 (dense #6, bm25 #6)
```
