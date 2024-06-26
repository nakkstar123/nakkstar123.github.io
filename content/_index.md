---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing


sections:
 - block: hero
   demo: true # Only display this section in the Hugo Blox Builder demo site
   content:
     title: Hugo Academic Theme
     image:
       filename: hero-academic.png
     cta:
       label: '**Get Started**'
       url: https://hugoblox.com/templates/
     cta_alt:
       label: Ask a question
       url: https://discord.gg/z8wNYzb
     cta_note:
       label: >-
         <div style="text-shadow: none;"><a class="github-button" href="https://github.com/HugoBlox/hugo-blox-builder" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star">Star Hugo Blox Builder</a></div><div style="text-shadow: none;"><a class="github-button" href="https://github.com/HugoBlox/theme-academic-cv" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star">Star the Academic template</a></div>
     text: |-
       **Generated by Hugo Blox Builder - the FREE, Hugo-based open source website builder trusted by 500,000+ sites.**


       **Easily build anything with blocks - no-code required!**


       From landing pages, second brains, and courses to academic resumés, conferences, and tech blogs.


       <!--Custom spacing-->
       <div class="mb-3"></div>
       <!--GitHub Button JS-->
       <script async defer src="https://buttons.github.io/buttons.js"></script>
   design:
     background:
       gradient_end: '#1976d2'
       gradient_start: '#004ba0'
       text_color_light: true
 - block: about.biography
   id: about
   content:
     title: About Me
     # Choose a user profile to display (a folder name within `content/authors/`)
     username: admin
#  - block: skills
#    content:
#      title: Skills
#      text: ''
#      # Choose a user to display skills from (a folder name within `content/authors/`)
#      username: admin
#    design:
#      columns: '1'
 - block: experience # add logos later
   id: exp
   content:
     title: Experience
     # Date format for experience
     #   Refer to https://docs.hugoblox.com/customization/#date-format
     date_format: Jan 2006
     # Experiences.
     #   Add/remove as many `experience` items below as you like.
     #   Required fields are `title`, `company`, and `date_start`.
     #   Leave `date_end` empty if it's your current employer.
     #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
     items:
       - title: President & Co-Founder
         company: Theory@UCLA
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2023-03-01'
         date_end: ''
         description: |2-
             * Established the first theoretical computer science student-run organization at UCLA and recruited over 40 graduate and undergraduate students
             * Organized quarterly reading groups to present topics in lattice cryptography, zero-knowledge proofs, secure computation, etc.
             * Hosted exam review sessions for Algorithms and Complexity [CS 180] and Theory of Computation [CS 181]
       - title: Course Reader | Theory of Computation [CS 181]
         company: UCLA Department of Computer Science
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2024-03-01'
         date_end: '2024-06-01'
         description: |2-
             * Graded weekly proof-based homeworks for theory of computation 
             * Provided 150+ students with detailed individual feedback 
       - title: Lead Instructor | Intermediate 2A
         company: Olga Radko Endowed Math Circle
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2023-06-01'
         date_end: ''
         description: |2-
             * Organized and led weekly math classes for 25 middle school and high school students
             * Designed worksheets in topics including graph theory, error-correcting codes, and combinatorics
       - title: Course Reader | Discrete Mathematics [Math 61]
         company: UCLA Department of Mathematics
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2024-01-01'
         date_end: '2024-04-01'
         description: |2-
             * Created rubrics and graded weekly proof-based homeworks for discrete mathematics
             * Provided 80+ students with detailed individual feedback 
       - title: Course Reader | Honors Linear Algebra [Math 115AH]
         company: UCLA Department of Mathematics
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2023-03-01'
         date_end: '2024-06-01'
         description: |2-
             * Created rubrics and graded weekly proof-based homeworks for honors linear algebra 
             * Provided 200+ students with detailed individual feedback 
       - title: Assistant Instructor | Advanced 3
         company: Olga Radko Endowed Math Circle
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2022-12-01'
         date_end: '2023-06-01'
         description: |2-
             * Assisted in teaching weekly math classes for 30 high school students
             * Worked solutions and helped students through worksheets in algebraic geometry
       - title: Lead Instructor | Advanced 2A
         company: Olga Radko Endowed Math Circle
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2022-06-01'
         date_end: '2022-12-01'
         description: |2-
             * Organized and led weekly math classes for 25 high school students
             * Designed worksheets in topics including special relativity and optimization
       - title: Lead Engineer, Trajectory Simulations | Prometheus Hybrid Rocket
         company: Rocket Project at UCLA
         company_url: ''
         company_logo: ''
         location: California
         date_start: '2021-12-01'
         date_end: '2022-06-01'
         description: |2-
             * Programmed a 6 degree-of-freedom simulation model in MATLAB for the hybrid fuel rocket team at UCLA
             * Simulation was used to predict rocket apogee, off-the-rail-speed, and stability during ascent
       
   design:
     columns: '2'

 - block: collection
   id: research 
   content:
     title: Research
     text: ''
     filters:
       folders:
         - publications
       exclude_featured: true
   design:
     columns: '2'
     view: compact

 - block: collection
   id: handouts
   content:
     title: Handouts
     # add theory scribes and latex lecture notes later as well
     text: |- 
          ## Olga Radko Endowed Math Circle Worksheets

          - <strong>[Optimization in Optics](https://circles.math.ucla.edu/circles/events.shtml?id=3370) - Nov 6, 2022 </strong>
          - <strong>[Optimization in Geometry](https://circles.math.ucla.edu/circles/events.shtml?id=3387) - Nov 20, 2022 </strong>
          - [Combinatorics](https://circles.math.ucla.edu/circles/events.shtml?id=3857) - Oct 8, 2023
          - [Induction](https://circles.math.ucla.edu/circles/events.shtml?id=3873) - Oct 15, 2023
          - <strong>[Error Correcting Codes](https://circles.math.ucla.edu/circles/events.shtml?id=3919) - Oct 29, 2023 </strong>
          - <strong>[Graph Games: Sim and Sprouts](https://circles.math.ucla.edu/circles/events.shtml?id=3961) - Nov 19, 2023 </strong>
          - [Reed Solomon Codes](https://circles.math.ucla.edu/circles/events.shtml?id=3992) - Dec 10, 2023 
          - <strong>[Information Theory](https://circles.math.ucla.edu/circles/events.shtml?id=4037) - Jan 14, 2024 </strong>
          - [Balanced Trees and Fibonacci Numbers](https://circles.math.ucla.edu/circles/events.shtml?id=4074) - Jan 28, 2024
          - <strong>[Counting Hats and Euler's number](https://circles.math.ucla.edu/circles/events.shtml?id=4115) - Feb 18, 2024 </strong>
          - [Finite Fields](https://circles.math.ucla.edu/circles/events.shtml?id=4225) - April 14, 2024
          - [Tiling and Coloring](https://circles.math.ucla.edu/circles/events.shtml?id=4302) - May 12, 2024

     filters:
       folders:
         - handouts
       exclude_featured: true
   design:
     columns: '2'
     view: compact 
 # - block: collection
 #   id: talks
 #   content:
 #     title: Recent & Upcoming Talks
 #     filters:
 #       folders:
 #         - event
 #   design:
 #     columns: '2'
 #     view: compact
 # - block: tag_cloud
 #   content:
 #     title: Popular Topics
 #   design:
 #     columns: '2'
 - block: contact
   id: contact
   content:
     title: Contact
     subtitle:
    #  text: |-
    #    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam mi diam, venenatis ut magna et, vehicula efficitur enim.
    #  # Contact (add or remove contact options as necessary)
     email: nakulkhambhati@g.ucla.edu
     phone: +1 341 500 3823
    #  appointment_url: 'https://calendly.com'
     address:
       street: 540 Midvale Ave
       city: Los Angeles
       region: CA
       postcode: '90024'
       country: United States
       country_code: US
    #  directions: Enter Building 1 and take the stairs to Office 200 on Floor 2
    #  office_hours:
    #    - 'Monday 10:00 to 13:00'
    #    - 'Wednesday 09:00 to 10:00'
     # Choose a map provider in `params.yaml` to show a map from these coordinates
     coordinates:
       latitude: '34.07'  
       longitude: '-118.4519' 
    #  contact_links:
    #    - icon: twitter
    #      icon_pack: fab
    #      name: DM Me
    #      link: 'https://twitter.com/Twitter'
    #    - icon: skype
    #      icon_pack: fab
    #      name: Skype Me
    #      link: 'skype:echo123?call'
    #    - icon: video
    #      icon_pack: fas
    #      name: Zoom Me
    #      link: 'https://zoom.com'
     # Automatically link email and phone or display as text?
     autolink: true
     # Email form provider
    #  form:
    #    provider: netlify
    #    formspree:
    #      id:
    #    netlify:
    #      # Enable CAPTCHA challenge to reduce spam?
    #      captcha: false
   design:
     columns: '2'
---


