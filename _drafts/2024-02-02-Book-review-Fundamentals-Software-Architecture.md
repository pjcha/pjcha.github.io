---
layout: post
title: 
date: 2024-02-02 00:20:00
description: Book review for Fundamentals of Software Architecture - Mark Richards and Neal Ford 
tags: Book-reviews, Architecture, Design
categories: Architecture books
chart:
  vega_lite: true
giscus_comments: true
toc:
  sidebar: left
---

# Why did I pick this book?
Coming from product design background and writing software in embedded my curiosity drives me to go beyond software design and development layers. Just like any hardware based product, software products are an amalgamation of different software principles. This natural curve 
opened opportunity for me to take on architectural tasks within the scope of my development activities. Offlately I have been getting involved at the system level components and piecing out the different information points into the system architecture. To take this further into a streamlined learning, it is impossible to not stumble upon this book.  

In the following part of this arcticle, I have tried to give you a brief introduction into the book and give you my personal learnings from it. This article is purely academic purpose and is part of my contribution to the software community world. If this interests you I would highly recommend to read the book further.

## Structure of the book
This book is written in three independent sections to cover the wholistic view on software architecture.  
*Part I*: Covers the foundation of the software architecture, defining it and breaking it down into it's core blocks. It also covers means of identifying, measuring and governing architectural characteristics.
*Part II*: Here it covers different archectural designs and accessment of the characteristics of it in star rating format.  
*Part III*: Then it talks about recording architectural decisions - on useful methods and tools to do it. The author also speaks about usefulness of soft skills to do the job of architect.  

Following are the key highlights I have noted and written in my words.

# What software architecture is about?
## Defining the architecture
On the same lines of Martin Fowler's definition in his famous article ["Who needs an architect"](<TBD>) quoted "Software Architecture is about the important stuff, what ever that is", the authors re-enforces this and has made an excellent attempt to explains us why that is true. In the age of shortly existed programming world, software architecture has a constantly evolving roles and hence it's definition. In the most general terms, it can be equivalent to undestanding the system which is divided into its components and connected with interfaces. As the authors quoted  
> It is the highest level concept of the expert developers and their shared understanding
The focus of software architecture should be more on the `WHY` than the `HOW` part.

## Software architecture's 4 sides
Th  e design process of a effective piece of software canv be  
* Structure
* Characteristics
* Decisions
* Design principles

#### Architectural thinking : broading the breadth
A developer's mindset is to dig deeper into few chosen / relevant topics, while an architect needs to have an overview on range of different technologies. Surely an architect needs to know few topics well in depth but the part where they start making the difference is having an overview of the technologies used in the product.

#### Practicing with Architectural Katas  
To become a good architect one needs to practice making architects designs. In the beginning of my journey into this I myself struggled to find platforms to excercise this. This book introduces the concept of architectural katas b originally formed by Ted Neward. It is an excercise where bunch of enthusiacts come together to design different systems and a moderator keeps track of time and assigns the tasks. I plan to organise such events in the near future, so if you are interested do drop me a message.  

### About architectural characteristics
#### Defining architectural characteristics  
There are two types of architecture characteristics - explict and implicit. Explicit characteristics are the ones which directly surfaces based on solution provided and are non-domain specific design consideration. On the other hand implicit characteristics are the ones which are influenced from the structural aspects of the designs. Let us take an example of design a website for online ticket booking system - here the characteristics such as reliability, availability, performance, scalibity, etc are explicitly covered while aspects such extensibility, upgradibility, privacy, security etc. I agree it is not straight forward to label the characteristics, but the important pov to keep is to think from the difference of non-domains or domain related.

#### Identifying architectural characteristics in a given use case
One of the initial step while architecting is to identify the characteristics. They can be extracted from the domain knowledge and the requirement documents. There is not specific set of rules to follow while indentifying them but one can practice through architectural kata excercises. 

#### Measuring characteristics



#### Governing architectural

#### Component based thinking - Conways Law

### Brief introductions into the architectural patterns
The author covered various architectural designs from definition to detailed description about them. The designs covered are the following - 
- Big ball of mud
- Monolithic architecture
- Component based design
- Servcie oriented architecture
- Service defined architecture
- Microservices
- Space based architecture

If you are interested do dive in to the details of it. The interesting part for me was the comparision of all designs. The above described architectural characteristics are used as the comparision point.

### Practices and soft skills

#### Decision making  
##### Importance of recording architectural records - adrtools  
##### Diagramming and capturing details  
#### Developing a Architectural path  
